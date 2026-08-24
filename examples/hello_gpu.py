"""Sanity check for a fresh Nautilus GPU pod: what card did I get, is it
exclusive, and does the hardware energy counter work?

    kubectl cp hello_gpu.py my-gpu-pod:/workspace/hello_gpu.py
    kubectl exec my-gpu-pod -- python /workspace/hello_gpu.py
"""
import pynvml

pynvml.nvmlInit()
h = pynvml.nvmlDeviceGetHandleByIndex(0)

name = pynvml.nvmlDeviceGetName(h)
mem = pynvml.nvmlDeviceGetMemoryInfo(h)
procs = pynvml.nvmlDeviceGetComputeRunningProcesses(h)
power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0

print(f"GPU        : {name}")
print(f"VRAM       : {mem.total / 2**30:.1f} GiB total, {mem.used / 2**30:.1f} used")
print(f"power now  : {power:.1f} W")
print(f"exclusive  : {len(procs) == 0}  ({len(procs)} other compute processes)")

try:
    e = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    print(f"energy ctr : OK ({e/1000:.0f} J since driver load) — trustworthy on "
          "datacenter cards; on consumer cards prefer integrating power samples")
except pynvml.NVMLError as ex:
    print(f"energy ctr : NOT available ({ex}) — measure energy as ∫ power·dt")

print("HELLO_GPU_DONE")
