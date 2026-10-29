"""(보너스 1) NumPy만으로 구현한 Gaussian Blur.

- OpenCV 없이 가우시안 커널 생성 + 컨볼루션을 직접 구현
- 이중 for문 버전 vs 벡터화(stride trick) 버전의 속도를 비교
"""

from __future__ import annotations

import time

import numpy as np


def gaussian_kernel(size: int = 5, sigma: float | None = None) -> np.ndarray:
    """정규화된 2D 가우시안 커널 생성."""
    if size % 2 == 0:
        size += 1
    if sigma is None:
        # OpenCV 기본식과 유사: sigma = 0.3*((size-1)*0.5 - 1) + 0.8
        sigma = 0.3 * ((size - 1) * 0.5 - 1) + 0.8
    ax = np.arange(size) - (size - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    return kernel / kernel.sum()


def _pad(gray: np.ndarray, pad: int) -> np.ndarray:
    return np.pad(gray, pad, mode="reflect")


def convolve_naive(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """이중 for문 기반 컨볼루션(이해용, 느림)."""
    k = kernel.shape[0]
    pad = k // 2
    padded = _pad(gray.astype(np.float32), pad)
    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float32)
    for i in range(h):
        for j in range(w):
            region = padded[i : i + k, j : j + k]
            out[i, j] = np.sum(region * kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def convolve_vectorized(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """stride trick으로 모든 윈도우를 한 번에 만들어 벡터 연산(빠름)."""
    k = kernel.shape[0]
    pad = k // 2
    padded = _pad(gray.astype(np.float32), pad)
    h, w = gray.shape
    # (h, w, k, k) 형태의 슬라이딩 윈도우 뷰
    windows = np.lib.stride_tricks.sliding_window_view(padded, (k, k))
    # windows shape: (h, w, k, k) -> 커널과 요소곱 후 합
    out = np.einsum("ijkl,kl->ij", windows, kernel)
    return np.clip(out, 0, 255).astype(np.uint8)


def benchmark(gray: np.ndarray, size: int = 5) -> dict[str, float]:
    """두 구현의 속도를 비교하고 결과가 일치하는지 확인한다."""
    kernel = gaussian_kernel(size)

    t0 = time.perf_counter()
    out_naive = convolve_naive(gray, kernel)
    t1 = time.perf_counter()
    out_vec = convolve_vectorized(gray, kernel)
    t2 = time.perf_counter()

    max_diff = int(np.abs(out_naive.astype(int) - out_vec.astype(int)).max())
    return {
        "naive_sec": t1 - t0,
        "vectorized_sec": t2 - t1,
        "speedup": (t1 - t0) / max(t2 - t1, 1e-9),
        "max_abs_diff": max_diff,
    }


if __name__ == "__main__":
    import argparse

    import cv2

    ap = argparse.ArgumentParser(description="NumPy Gaussian Blur 벤치마크")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=5)
    args = ap.parse_args()

    data = np.fromfile(args.image, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    # 벤치마크가 과도하게 느리지 않도록 축소
    img = cv2.resize(img, (320, 240))

    stats = benchmark(img, args.size)
    print(f"naive      : {stats['naive_sec']*1000:8.2f} ms")
    print(f"vectorized : {stats['vectorized_sec']*1000:8.2f} ms")
    print(f"speedup    : {stats['speedup']:8.1f} x")
    print(f"max_abs_diff(두 구현 차이): {stats['max_abs_diff']}")
