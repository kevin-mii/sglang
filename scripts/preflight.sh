# Fail loudly before a 10-minute server boot if the client cannot run.
# The uv-managed CPython 3.12 under /root/.local/share/uv/python was deleted
# between sessions, which left aiperf-venv's python symlink dangling and cost
# us one arm (exit 126 after a full boot). Verify, do not assume.
set -u
if ! /scratch/agentx/aiperf-venv/bin/aiperf --version >/dev/null 2>&1; then
  echo "PREFLIGHT FAILED: aiperf not runnable"
  /scratch/agentx/aiperf-venv/bin/aiperf --version 2>&1 | tail -3
  exit 1
fi
echo "preflight ok: aiperf $(/scratch/agentx/aiperf-venv/bin/aiperf --version 2>&1 | tail -1)"
