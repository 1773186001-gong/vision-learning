
import torch
import time


def main():
    print("=" * 50)
    print("RTX 3060 GPU Server Test")
    print("=" * 50)

    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable.")
        return

    device = torch.device("cuda:0")
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", round(
        torch.cuda.get_device_properties(0).total_memory / 1024**3, 2
    ), "GB")

    start = time.time()

    # 在 GPU 上创建张量并执行矩阵乘法
    x = torch.randn(2048, 2048, device=device)
    y = x @ x

    torch.cuda.synchronize()
    elapsed = time.time() - start

    print("Tensor device:", y.device)
    print("Tensor shape:", tuple(y.shape))
    print("Results finite:", torch.isfinite(y).all().item())
    print("Allocated VRAM:", round(
        torch.cuda.memory_allocated() / 1024**2, 1
    ), "MB")
    print("Elapsed time:", round(elapsed, 4), "seconds")
    print("GPU test passed!")


if __name__ == "__main__":
    main()
