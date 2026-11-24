#!/usr/bin/env bash
# Serialize every NPU job in this container. The lock is held by the open file
# descriptor and is released by the kernel only after the whole exec'd job exits.
# Unlike a pgrep check, two starters cannot both pass before either child exists.

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PER_TEST_JOB_LOCK="${TMPDIR:-/tmp}/vllm-ascend-per-test-npu.lock"

acquire_per_test_job_lock() {
    local job="${1:-unknown}"
    if ! command -v flock >/dev/null 2>&1; then
        echo "❌ 缺少 flock，无法保证 NPU 测试互斥；请安装 util-linux 后重试。" >&2
        return 1
    fi

    # Bash dynamically allocates a descriptor >= 10. Descriptors opened this way
    # survive exec, so run.sh -> sweep_cases.sh -> python all keep the same lock.
    exec {PER_TEST_JOB_LOCK_FD}>>"$PER_TEST_JOB_LOCK" || {
        echo "❌ 无法打开测试锁：$PER_TEST_JOB_LOCK" >&2
        return 1
    }
    if ! flock -n "$PER_TEST_JOB_LOCK_FD"; then
        echo "❌ 已有一趟 per_test 在跑，已拒绝叠跑（不会向旧任务发送任何信号）。" >&2
        echo "   持锁信息：$(cat "$PER_TEST_JOB_LOCK" 2>/dev/null || echo '<不可读>')" >&2
        echo "   查看状态：python3 \"$HERE/job_control.py\" status" >&2
        echo "   安全停止：bash \"$HERE/run.sh stop\"（先 SIGTERM，绝不自动 SIGKILL）" >&2
        eval "exec ${PER_TEST_JOB_LOCK_FD}>&-"
        return 1
    fi

    # Only the lock owner may rewrite this diagnostic file. The lock is held
    # by the still-open descriptor even while the file text is replaced.
    printf 'pid=%s started=%s job=%s cwd=%s\n' \
        "$$" "$(date -Is)" "$job" "$PWD" >"$PER_TEST_JOB_LOCK"
    export PER_TEST_LOCK_HELD=1
    echo "🔒 已取得 NPU 测试互斥锁：$PER_TEST_JOB_LOCK（pid=$$, job=$job）"
}
