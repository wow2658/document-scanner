"""이미지 전처리 및 엣지 검출 단계.

파이프라인: Grayscale -> Gaussian Blur -> Canny Edge
각 단계는 독립 함수로 분리하여 단계별 시각화/실험이 가능하도록 설계했다.
"""

from __future__ import annotations

import cv2
import numpy as np


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """컬러(BGR) 이미지를 Grayscale로 변환한다. 이미 1채널이면 그대로 반환."""
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def odd_kernel(size: int) -> int:
    """가우시안 커널 크기는 홀수여야 한다. 짝수/0이 들어오면 보정한다."""
    size = int(size)
    if size < 1:
        return 1
    if size % 2 == 0:
        size += 1
    return size


def gaussian_blur(gray: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """가우시안 블러로 노이즈를 제거한다.

    kernel_size: 홀수여야 하며, 값이 클수록 더 강하게 평활화된다.
    """
    k = odd_kernel(kernel_size)
    return cv2.GaussianBlur(gray, (k, k), 0)


def canny_edge(blurred: np.ndarray, low: int = 50, high: int = 150) -> np.ndarray:
    """Canny 엣지 검출.

    low, high: 이력(hysteresis) 임계값. 일반적으로 high ≈ 2~3 * low 를 권장한다.
    """
    low = max(int(low), 0)
    high = max(int(high), low + 1)  # high > low 보장
    return cv2.Canny(blurred, low, high)


def preprocess_for_edges(
    image: np.ndarray,
    blur_kernel: int = 5,
    canny_low: int = 50,
    canny_high: int = 150,
) -> dict[str, np.ndarray]:
    """원본 -> gray -> blur -> edge 중간 산출물을 한 번에 반환한다.

    Returns
    -------
    dict: {"gray", "blurred", "edged"}
    """
    gray = to_grayscale(image)
    blurred = gaussian_blur(gray, blur_kernel)
    edged = canny_edge(blurred, canny_low, canny_high)
    # 끊어진 엣지를 이어 윤곽선 검출률을 높인다.
    edged = cv2.dilate(edged, np.ones((3, 3), np.uint8), iterations=1)
    return {"gray": gray, "blurred": blurred, "edged": edged}
