"""1-minute Modal GPU smoke test for sheet-music-converter OMR."""

import time
import modal

app = modal.App("omr-gpu-smoke-test")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("torch", index_url="https://download.pytorch.org/whl/cu126")
)


@app.function(gpu="T4", timeout=120, image=image)
def smoke_gpu() -> dict:
    import torch

    cuda_avail = torch.cuda.is_available()
    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": cuda_avail,
    }
    if not cuda_avail:
        print("CUDA NOT AVAILABLE", info)
        return info

    props = torch.cuda.get_device_properties(0)
    info["gpu_name"] = torch.cuda.get_device_name(0)
    info["vram_gb"] = round(props.total_memory / (1024**3), 2)
    info["compute_capability"] = f"{props.major}.{props.minor}"
    info["bf16_supported"] = bool(torch.cuda.is_bf16_supported())
    info["fp16_supported"] = True

    # ~1-second FP16 matmul benchmark
    n = 4096
    a = torch.randn((n, n), device="cuda", dtype=torch.float16)
    b = torch.randn((n, n), device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()

    iters = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        c = a @ b
        iters += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    tflops = (2.0 * n * n * n * iters) / dt / 1e12

    info["matmul_iters"] = iters
    info["matmul_sec"] = round(dt, 3)
    info["fp16_tflops"] = round(tflops, 2)
    info["checksum"] = round(float(c[0, 0].item()), 4)

    for k, v in info.items():
        print(f"{k}: {v}")
    return info


@app.local_entrypoint()
def main():
    t0 = time.perf_counter()
    res = smoke_gpu.remote()
    wall = round(time.perf_counter() - t0, 2)
    print(f"local_wall_sec: {wall}")
    print(f"result: {res}")
