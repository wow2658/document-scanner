"""기하학적 변환에 필요한 좌표 정렬 로직.

요구사항:
    "좌표 정렬 로직은 라이브러리에 의존하지 않고 기하학적 원리로 직접 구현한다."

따라서 cv2.minAreaRect 등 외부 함수의 꼭짓점 정렬 기능을 쓰지 않고,
좌표의 합(x+y)과 차(y-x)라는 기하학적 성질만으로 순서를 정한다.
"""

from __future__ import annotations

import numpy as np


def order_points(pts: np.ndarray) -> np.ndarray:
    """4개의 (x, y) 좌표를 [좌상, 우상, 우하, 좌하] 순서로 정렬한다.

    기하학적 원리
    --------------
    이미지 좌표계는 오른쪽으로 갈수록 x가 커지고, 아래로 갈수록 y가 커진다.

    1) 좌표의 합 s = x + y
       - 좌상단은 x도 작고 y도 작으므로 s가 **최소**
       - 우하단은 x도 크고 y도 크므로 s가 **최대**
    2) 좌표의 차 d = y - x
       - 우상단은 x가 크고 y가 작으므로 d가 **최소**
       - 좌하단은 x가 작고 y가 크므로 d가 **최대**

    Parameters
    ----------
    pts : np.ndarray, shape (4, 2)
        정렬되지 않은 4개의 좌표.

    Returns
    -------
    np.ndarray, shape (4, 2), dtype float32
        [좌상(TL), 우상(TR), 우하(BR), 좌하(BL)] 순서의 좌표.
    """
    pts = np.asarray(pts, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)          # x + y
    d = pts[:, 1] - pts[:, 0]    # y - x

    ordered[0] = pts[np.argmin(s)]  # 좌상 (합 최소)
    ordered[2] = pts[np.argmax(s)]  # 우하 (합 최대)
    ordered[1] = pts[np.argmin(d)]  # 우상 (차 최소)
    ordered[3] = pts[np.argmax(d)]  # 좌하 (차 최대)

    return ordered


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """두 점 사이의 유클리드 거리."""
    return float(np.sqrt(((a - b) ** 2).sum()))


def target_size(ordered: np.ndarray) -> tuple[int, int]:
    """정렬된 4점으로부터 결과 이미지의 (너비, 높이)를 계산한다.

    (보너스 2) 종횡비 자동 보정:
        원본 네 변의 실제 픽셀 길이를 측정하여 결과물이 찌그러지지 않도록
        가로/세로 크기를 결정한다.
    """
    tl, tr, br, bl = ordered

    width_top = euclidean(tr, tl)
    width_bottom = euclidean(br, bl)
    max_width = max(int(round(width_top)), int(round(width_bottom)))

    height_left = euclidean(bl, tl)
    height_right = euclidean(br, tr)
    max_height = max(int(round(height_left)), int(round(height_right)))

    # 0 나눗셈/퇴화 방지
    return max(max_width, 1), max(max_height, 1)
