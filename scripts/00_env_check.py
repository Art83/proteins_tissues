import json
import platform
import socket
import statistics
import sys
import time
from datetime import datetime
from importlib.metadata import version


import numpy as np
import torch

from configs import RESULTS

SEED = 1
MATMUL_SIZE = 4096
N_REPEATS = 5
PACKAGES = ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy", "statsmodels",
            "transformers", "huggingface_hub", "anndata", "h5py", "matplotlib", "torch"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
machine = socket.gethostname()

# 1. environment record
env = {
    "machine": machine,
    "run_time": datetime.now().isoformat(timespec="seconds"),
    "os": platform.platform(),
    "python": sys.version.split()[0],
    "torch": torch.__version__,
    "torch_cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    "gpu_memory_gib": round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1) if torch.cuda.is_available() else None,
    "cpu_threads": torch.get_num_threads(),
    "selected_device": str(device),
}
for pkg in PACKAGES:
    env[pkg] = version(pkg)

print("== 1. environment")
for key, value in env.items():
    print(f"{key:>18}: {value}")

# 2. tensors, dtypes, devices
print("\n== 2. tensors")
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
print("torch.tensor      shape", tuple(a.shape), "dtype", a.dtype, "device", a.device)

b = torch.from_numpy(np.array([1.0, 2.0]))
print("from numpy        dtype", b.dtype, "(numpy default carries over)")

a_dev = a.to(device)
print("after .to(device) device", a_dev.device)

print("float32 + float64 ->", (torch.ones(2, dtype=torch.float32) + torch.ones(2, dtype=torch.float64)).dtype)

if device.type == "cuda":
    try:
        a + a_dev
    except RuntimeError as err:
        print("mixing devices RuntimeError:", str(err).splitlines()[0])
else:
    print("mixing devices skipped, no GPU on this machine")

# 3. automatic differentiation
print("\n== 3. autograd")
torch.manual_seed(SEED)
X = torch.randn(5, 3, dtype=torch.float64)
y = torch.randn(5, dtype=torch.float64)
w = torch.randn(3, dtype=torch.float64, requires_grad=True)

loss = ((X @ w - y) ** 2).sum()
loss.backward()

grad_by_hand = 2 * X.T @ (X @ w.detach() - y)
print("autograd gradient ", w.grad.numpy().round(6))
print("hand-derived      ", grad_by_hand.numpy().round(6))
print("max abs difference", (w.grad - grad_by_hand).abs().max().item())

# 4. timing a matrix multiplication
print("\n== 4. timing", f"{MATMUL_SIZE}x{MATMUL_SIZE} float32 matmul, median of {N_REPEATS}")
timings = {}
devices = [torch.device("cpu")] + ([torch.device("cuda")] if torch.cuda.is_available() else [])
for dev in devices:
    m = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device=dev)
    m @ m
    if dev.type == "cuda":
        torch.cuda.synchronize()
        start = time.perf_counter()
        m @ m
        unsynced = time.perf_counter() - start
        torch.cuda.synchronize()
        print(f"cuda without synchronize: {unsynced:.5f} s")
    times = []
    for _ in range(N_REPEATS):
        start = time.perf_counter()
        m @ m
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    timings[dev.type] = statistics.median(times)
    print(f"{dev.type}: {timings[dev.type]:.5f} s")

if "cuda" in timings:
    print(f"cpu / cuda ratio: {timings['cpu'] / timings['cuda']:.1f}")

# 5. random seeds
print("\n== 5. seeds")
torch.manual_seed(SEED)
r1 = torch.rand(3)
torch.manual_seed(SEED)
r2 = torch.rand(3)
print("same seed, same draws:", torch.equal(r1, r2))

out = {
    "environment": env,
    "autograd_max_abs_diff": (w.grad - grad_by_hand).abs().max().item(),
    "matmul_median_seconds": timings,
    "seed_reproducible": torch.equal(r1, r2),
}
out_path = RESULTS / f"00_env_{machine}.json"
out_path.write_text(json.dumps(out, indent=2))
print("\nwritten", out_path)