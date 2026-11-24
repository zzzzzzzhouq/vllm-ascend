#!/usr/bin/env python3
"""Inspect or gracefully stop only this checkout's per_test processes.

Stopping never sends SIGKILL: a process stuck in the Ascend driver's SVM unmap
path cannot act on it, and forcing another cleanup attempt can make recovery
harder. SIGTERM is translated to KeyboardInterrupt by the bench/pytest code so
Python ``finally`` blocks can synchronize the NPU and uninitialize MemFabric.
"""

import argparse
import os
import pathlib
import signal
import subprocess
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
# Match on the repo root, not just per_test/: scripts like sweep_cases.sh are
# invoked via "bash per_test/sweep_cases.sh" relative to the repo, so their
# cmdline may not contain the absolute per_test path.
REPO = HERE.parent
POLL_SECONDS = 1.0
DEFAULT_STOP_TIMEOUT = 60.0
JOB_NAMES = (
    "sweep_cases.sh",
    "sweep_small.sh",
    "precision_small.sh",
    "sweep_median.py",
    "bench_fuse_sparse_attention_overlap.py",
    "test_fsao_precision.py",
)
LEAF_NAMES = (
    "bench_fuse_sparse_attention_overlap.py",
    "test_fsao_precision.py",
)


def _proc_dirs():
    proc = pathlib.Path("/proc")
    if not proc.is_dir():
        raise SystemExit("这个工具需要在 Linux /proc 下运行")
    for entry in proc.iterdir():
        if entry.name.isdigit():
            yield entry


def _cmdline(pid_dir):
    try:
        raw = (pid_dir / "cmdline").read_bytes()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return ""
    return raw.replace(b"\0", b" ").decode(errors="replace").strip()


def _proc_cwd(pid_dir):
    try:
        return pathlib.Path(os.readlink(pid_dir / "cwd"))
    except (FileNotFoundError, PermissionError, ProcessLookupError, OSError):
        return None


def _matches(pid_dir, cmdline):
    """A job belongs to this checkout iff it runs our scripts FROM our repo."""
    if not any(name in cmdline for name in JOB_NAMES):
        return False
    if str(HERE) in cmdline:
        return True
    # Relative invocation like "bash per_test/sweep_cases.sh": check cwd.
    cwd = _proc_cwd(pid_dir)
    if cwd is None:
        return False
    try:
        cwd.relative_to(REPO)
        return True
    except ValueError:
        return False


def _jobs():
    jobs = {}
    for pid_dir in _proc_dirs():
        pid = int(pid_dir.name)
        if pid in (os.getpid(), os.getppid()):
            continue
        cmdline = _cmdline(pid_dir)
        if _matches(pid_dir, cmdline):
            jobs[pid] = cmdline
    return jobs


def _status(pid):
    try:
        fields = (pathlib.Path("/proc") / str(pid) / "status").read_text().splitlines()
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return "?"
    for line in fields:
        if line.startswith("State:"):
            return line.split(":", 1)[1].strip()
    return "?"


def _alive(pid):
    state = _status(pid)
    return state != "?" and not state.startswith("Z")


def _diagnose(pid, cmdline):
    print(f"\n--- PID {pid}: {_status(pid)} ---", flush=True)
    print(f"cmdline: {cmdline}", flush=True)
    try:
        status = (pathlib.Path("/proc") / str(pid) / "status").read_text().splitlines()
        for line in status:
            if line.startswith(("State:", "SigPnd:", "SigBlk:", "SigCgt:")):
                print(line, flush=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return
    try:
        stack = (pathlib.Path("/proc") / str(pid) / "stack").read_text()
        print("kernel stack:\n" + stack, flush=True)
    except (FileNotFoundError, PermissionError, ProcessLookupError) as exc:
        print(f"kernel stack unavailable: {exc}", flush=True)
    if _status(pid).startswith("D"):
        print(
            "⚠ D 态不能靠 kill 清理。若栈包含 svm_occupy_da/devmm，"
            "停止反复强杀，保留现场并联系运维复位/重启。",
            flush=True,
        )


def _npu_processes():
    """Best-effort residue report; this command is diagnostic only."""
    visible = os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "0").split(",", 1)[0].strip()
    if not visible.isdigit():
        visible = "0"
    try:
        result = subprocess.run(
            ["npu-smi", "info", "-t", "proc", "-i", visible],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        print("\n===== npu-smi 本卡进程（用于核对残留）=====", flush=True)
        print((result.stdout or result.stderr).strip(), flush=True)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"npu-smi 残留检查不可用：{exc}", flush=True)


def status(_args):
    jobs = _jobs()
    if not jobs:
        print("✅ 当前 checkout 没有 per_test 进程")
        _npu_processes()
        return 0
    unhealthy = False
    for pid, cmdline in sorted(jobs.items()):
        state = _status(pid)
        print(f"PID {pid:>7}  {state:<24}  {cmdline}")
        if state.startswith("D"):
            unhealthy = True
            _diagnose(pid, cmdline)
    if not unhealthy:
        _npu_processes()
    return 1 if unhealthy else 0


def stop(args):
    jobs = _jobs()
    if not jobs:
        print("✅ 没有需要停止的 per_test 进程")
        _npu_processes()
        return 0

    # SIGTERM the innermost command (bench/pytest) first. Its parents
    # (sweep_median.py -> sweep_cases.sh -> run.sh) then see a normal non-zero
    # child and can tear down cleanly instead of dying mid-cleanup.
    leaves = {pid: cmd for pid, cmd in jobs.items() if any(name in cmd for name in LEAF_NAMES)}
    targets = leaves or jobs
    wrapper_pids = set(jobs) - set(targets)
    if wrapper_pids:
        print("等待叶进程退出后再由父脚本收尾：" + " ".join(map(str, sorted(wrapper_pids))), flush=True)
    for pid, cmdline in sorted(targets.items()):
        print(f"SIGTERM -> PID {pid}: {cmdline}", flush=True)
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    deadline = time.monotonic() + args.timeout
    while time.monotonic() < deadline:
        if not any(_alive(pid) for pid in targets):
            print("✅ 目标进程已退出；等待父脚本完成收尾")
            tail_deadline = time.monotonic() + 5.0
            while time.monotonic() < tail_deadline and any(_alive(pid) for pid in wrapper_pids):
                time.sleep(POLL_SECONDS)
            _npu_processes()
            return 0
        time.sleep(POLL_SECONDS)

    print(f"\n❌ {args.timeout:g}s 后仍有进程未退出；不会升级 SIGKILL。", flush=True)
    for pid, cmdline in sorted(targets.items()):
        if _alive(pid):
            _diagnose(pid, cmdline)
    # Wait for wrappers to see the leaf's exit and finish their own teardown.
    tail_deadline = time.monotonic() + 5.0
    while time.monotonic() < tail_deadline and any(_alive(pid) for pid in wrapper_pids):
        time.sleep(POLL_SECONDS)
    _npu_processes()
    print("请保留以上诊断；原任务仍持有互斥锁，新任务不会叠跑。", flush=True)
    return 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    status_parser = sub.add_parser("status", help="列出本 checkout 的任务并告警 D 态")
    status_parser.set_defaults(func=status)
    stop_parser = sub.add_parser("stop", help="SIGTERM 后等待，不做 SIGKILL")
    stop_parser.add_argument("--timeout", type=float, default=DEFAULT_STOP_TIMEOUT)
    stop_parser.set_defaults(func=stop)
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
