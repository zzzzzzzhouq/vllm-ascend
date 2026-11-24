#!/usr/bin/env bash
# A5 MemFabric VMM offload diagnostic collector (READ-ONLY by default).
#
# Goal: explain `HYBM hybm_vmm_based_segment.cpp AllocLocalMemory: Map mem failed:-6`
# on Atlas A5 (ascend950pr_*) when HugePages_Free is already 1024 x 2M.
#
# It does NOT modify hugepages, kernel params, registers or services.
# The only optional action is running a minimal MemFabric init probe (allocates
# and immediately frees a 1 GiB host pool).
#
# Usage:
#   bash a5_vmm_diag.sh                 # collect environment only
#   RUN_PROBE=1 bash a5_vmm_diag.sh     # + offload init probe (VMM, 1 GiB, no empty)
#   RUN_PROBE=alloc bash a5_vmm_diag.sh # + offload init + offload.empty(1 GiB)
#   RUN_PROBE=giant bash a5_vmm_diag.sh # + conn-based fallback probe
#   sudo -E RUN_PROBE=alloc bash a5_vmm_diag.sh
#
# The probe needs the same Python env as per_test (torch, torch_npu,
# memfabric_hybrid). Run from the repo or anywhere; the probe path is resolved
# relative to this script. Put a5_offload_probe.py next to this script.
#
# Output: ./a5_vmm_diag_YYYYmmdd_HHMMSS.log (also printed to stdout).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
LOG="${PWD}/a5_vmm_diag_${TS}.log"
PROBE="${SCRIPT_DIR}/a5_offload_probe.py"

log() { printf '\n===== %s =====\n' "$*"; }
run() {
    # run <label> <command...>
    log "$1"; shift
    "$@" 2>&1 || echo "[exit=$?] command failed: $*"
}
note() { printf '%s\n' "$*"; }

{
log "a5_vmm_diag start"
date -Is
hostname
whoami
pwd
uname -a
cat /etc/os-release 2>/dev/null

log "kernel cmdline (iomem/relaxed/huge/numa)"
cat /proc/cmdline
tr ' ' '\n' < /proc/cmdline | grep -iE 'iomem|relaxed|huge|numa|default_hugepagesz' || true

log "lscpu"
lscpu 2>/dev/null | head -40

log "numactl -H"
command -v numactl >/dev/null 2>&1 && numactl -H || echo "numactl not installed"

log "free -h"
free -h

log "memory / hugepage summary"
grep -iE 'MemTotal|MemFree|MemAvailable|CommitLimit|Committed_AS|HugePages|Hugepagesize|Hugetlb' /proc/meminfo

log "per-node hugepages (2M and 1G)"
for d in /sys/devices/system/node/node*/hugepages/hugepages-*; do
    [ -d "$d" ] || continue
    printf '%s total=%s free=%s resv=%s surp=%s\n' \
        "$d" \
        "$(cat "$d"/nr_hugepages 2>/dev/null)" \
        "$(cat "$d"/free_hugepages 2>/dev/null)" \
        "$(cat "$d"/resv_hugepages 2>/dev/null)" \
        "$(cat "$d"/surplus_hugepages 2>/dev/null)"
done

log "global hugepage sysctls"
cat /proc/sys/vm/nr_hugepages 2>/dev/null
cat /proc/sys/vm/nr_overcommit_hugepages 2>/dev/null
cat /proc/sys/vm/overcommit_memory 2>/dev/null
cat /proc/sys/vm/overcommit_ratio 2>/dev/null
cat /proc/sys/vm/max_map_count 2>/dev/null

log "transparent hugepage"
cat /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null
cat /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null

log "memlock limits"
ulimit -a
grep -iE 'Max locked memory|Max address space|Max map count' /proc/self/limits 2>/dev/null
sysctl -a 2>/dev/null | grep -iE 'memlock|hugetlb|hugepage' | head -40

log "buddyinfo (fragmentation; order-9 = 2M on 4K pages)"
cat /proc/buddyinfo 2>/dev/null
head -40 /proc/pagetypeinfo 2>/dev/null

log "NPU info"
npu-smi info 2>&1
npu-smi info -l 2>&1
npu-smi info -t board -i 0 2>&1 | head -40

log "driver / CANN / memfabric versions"
cat /usr/local/Ascend/driver/version.info 2>/dev/null
cat /usr/local/Ascend/ascend-toolkit/latest/version.cfg 2>/dev/null
ls /usr/local/Ascend 2>/dev/null
python3 - <<'PY' 2>&1
import os, sys
print("python", sys.version.replace("\n", " "))
for mod in ("torch", "torch_npu", "memfabric_hybrid"):
    try:
        m = __import__(mod)
        print(mod, getattr(m, "__version__", "<no __version__>"), getattr(m, "__file__", "<no file>"))
    except Exception as exc:
        print(mod, "IMPORT FAILED:", repr(exc))
try:
    import importlib.metadata as md
    print("memfabric_hybrid dist:", md.version("memfabric_hybrid"))
except Exception as exc:
    print("memfabric_hybrid dist: unknown", repr(exc))
PY

log "memfabric libraries"
find / -name 'libmf_hybm*.so' -o -name 'libmf_*.so' 2>/dev/null | head -20
command -v mfcli >/dev/null 2>&1 && mfcli kernel info 2>&1 || echo "mfcli not found"

log "device nodes / kernel modules"
ls -l /dev/davinci* /dev/davinci_manager /dev/hisi_hdc /dev/devmm_svm 2>&1 | head -30
lsmod 2>/dev/null | grep -iE 'davinci|ascend|hybm|svm' || true

log "relevant environment variables"
env | grep -E '^(MF_|MEMFABRIC|DSA_|DRAM_GB|ASCEND|SOC_VERSION|LD_LIBRARY_PATH|PYTHONPATH|HCCL_)' | sort

log "A5 VMM registers (read-only; expected A5 doc value: 0x0F00807F)"
if command -v busybox >/dev/null 2>&1; then
    echo -n "0x3340004c1c = "; busybox devmem 0x3340004c1c 32 2>&1 || echo "<read failed>"
    echo -n "0x7340004c1c = "; busybox devmem 0x7340004c1c 32 2>&1 || echo "<read failed>"
else
    echo "busybox not found; cannot read registers"
fi

log "kernel log hints (dmesg tail, may need root)"
dmesg 2>/dev/null | grep -iE 'hybm|vmm|hugetlb|hugepage|davinci|ascend|iommu|out of memory|oom' | tail -200 || echo "dmesg unavailable"

log "recent per_test logs (if any)"
for f in $(ls -t per_test/logs/*.log 2>/dev/null | head -3); do
    echo "--- $f"
    grep -nE 'Try HalMemCreate|HalMemMap|HalMemAddress|HYBM|OFFLOAD|huge|numa|get_user|offload_init' "$f" | tail -80 || true
done

case "${RUN_PROBE:-0}" in
    0|"")
        log "offload probe"
        note "skipped (set RUN_PROBE=1|alloc|giant to run ${PROBE})"
        ;;
    *)
        if [ ! -f "$PROBE" ]; then
            log "offload probe"
            note "probe not found: $PROBE"
        else
            PROBE_ARGS=(--gb 1)
            case "${RUN_PROBE}" in
                alloc) PROBE_ARGS+=(--alloc) ;;
                giant) PROBE_ARGS+=(--giant-page --alloc) ;;
                *)     ;;
            esac
            log "offload probe: python3 $PROBE ${PROBE_ARGS[*]}"
            python3 "$PROBE" "${PROBE_ARGS[@]}" 2>&1
            note "[probe exit=$?]"
        fi
        ;;
esac

log "a5_vmm_diag end"
note "Attach this log plus the failing per_test log. Do not edit registers/hugepages"
note "until the read-only facts above are reviewed."
} | tee "$LOG"

echo
echo "saved: $LOG"
