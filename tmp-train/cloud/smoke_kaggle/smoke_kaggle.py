"""1-minute Kaggle GPU smoke test for sheet-music-converter OMR."""

import platform
import subprocess
import time
import torch

print("=== SYSTEM ===")
print("python_version:", platform.python_version())
print("torch_version:", torch.__version__)
print("cuda_version:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

print("\n=== NVIDIA-SMI ===")
r = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
print(r.stdout)
if r.stderr:
    print("nvidia-smi stderr:", r.stderr)

if torch.cuda.is_available():
    for idx in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(idx)
        print(
            f"gpu[{idx}]: {torch.cuda.get_device_name(idx)} | "
            f"vram_gb={props.total_memory / (1024**3):.2f} | "
            f"cc={props.major}.{props.minor}"
        )
    print("bf16_supported:", bool(torch.cuda.is_bf16_supported()))

    # ~1-second FP16 matmul benchmark on GPU 0
    n = 4096
    a = torch.randn((n, n), device="cuda:0", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda:0", dtype=torch.float16)
    torch.cuda.synchronize()

    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        c = a @ b
        iters += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = (2.0 * n * n * n * iters) / dt / 1e12
    print(
        f"matmul_fp16: iters={iters} sec={dt:.3f} tflops={tflops:.2f} "
        f"checksum={float(c[0, 0].item()):.4f}"
    )
else:
    print("ERROR: CUDA not available on Kaggle GPU kernel!")
