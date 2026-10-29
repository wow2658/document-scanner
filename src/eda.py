"""이미지 탐색적 분석 (EDA).

요구사항:
    - 각 이미지의 픽셀 히스토그램 분석, 밝기 분포 특성 파악
    - 쉬운 케이스 vs 어려운 케이스의 히스토그램 차이 시각화
    - "왜 어두운 이미지에서 엣지 검출이 어려운지"를 분포 관점에서 설명하기 위한 지표 산출
"""

from __future__ import annotations

import glob
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

from .preprocess import to_grayscale


def brightness_stats(gray: np.ndarray) -> dict[str, float]:
    """밝기 분포의 요약 통계.

    - mean/std : 전반적 밝기와 대비(표준편차가 작을수록 저대비 → 엣지 약함)
    - p5/p95   : 하위/상위 5% 분위 (동적 범위)
    - dark_ratio : 어두운 픽셀(<50) 비율 (높을수록 저조도)
    """
    g = gray.astype(np.float32)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "p5": float(np.percentile(g, 5)),
        "p95": float(np.percentile(g, 95)),
        "dark_ratio": float((gray < 50).mean()),
        "bright_ratio": float((gray > 205).mean()),
    }


def plot_histograms(
    easy_paths: list[str],
    hard_paths: list[str],
    out_path: str = "report/figures/eda_histograms.png",
) -> str:
    """쉬운/어려운 케이스의 평균 밝기 히스토그램을 겹쳐 비교 저장한다."""

    def _avg_hist(paths: list[str]) -> np.ndarray | None:
        hists = []
        for p in paths:
            data = np.fromfile(p, dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is None:
                continue
            gray = to_grayscale(img)
            h = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
            hists.append(h / h.sum())  # 정규화(이미지 크기 영향 제거)
        if not hists:
            return None
        return np.mean(hists, axis=0)

    easy = _avg_hist(easy_paths)
    hard = _avg_hist(hard_paths)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.figure(figsize=(9, 5))
    if easy is not None:
        plt.plot(easy, label=f"Easy (n={len(easy_paths)})", color="tab:blue")
    if hard is not None:
        plt.plot(hard, label=f"Hard (n={len(hard_paths)})", color="tab:red")
    plt.title("Average Brightness Histogram: Easy vs Hard")
    plt.xlabel("Pixel intensity (0=dark, 255=bright)")
    plt.ylabel("Normalized frequency")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120)
    plt.close()
    return out_path


def run_eda(image_dir: str, out_path: str = "report/figures/eda_histograms.png") -> None:
    """디렉터리의 이미지를 파일명 규칙으로 분류해 EDA를 수행한다.

    파일명에 'easy'가 포함되면 쉬운 케이스, 그 외는 어려운 케이스로 간주한다.
    각 이미지의 밝기 통계도 함께 출력한다.
    """
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not paths:
        print(f"[경고] 이미지가 없습니다: {image_dir}")
        return

    easy = [p for p in paths if "easy" in os.path.basename(p).lower()]
    hard = [p for p in paths if p not in easy]

    print(f"{'파일':40s} {'mean':>7s} {'std':>7s} {'dark%':>7s}")
    for p in paths:
        data = np.fromfile(p, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            continue
        s = brightness_stats(to_grayscale(img))
        print(
            f"{os.path.basename(p):40s} "
            f"{s['mean']:7.1f} {s['std']:7.1f} {s['dark_ratio']*100:6.1f}%"
        )

    saved = plot_histograms(easy, hard, out_path)
    print(f"\n[저장됨] 히스토그램 비교: {saved}")
    print(
        "해석 힌트: 'std'(대비)가 작고 'dark%'가 높은 이미지일수록 "
        "엣지의 밝기 경사가 완만해 Canny 엣지 검출이 어려워진다."
    )


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="이미지 EDA (밝기 히스토그램)")
    ap.add_argument("--dir", default="images", help="이미지 디렉터리")
    ap.add_argument("--out", default="report/figures/eda_histograms.png")
    args = ap.parse_args()
    run_eda(args.dir, args.out)
