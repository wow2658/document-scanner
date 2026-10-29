"""문서 스캐너 핵심 파이프라인.

단계: 전처리 -> 엣지 검출 -> 윤곽선 추출 -> 사각형 검출 -> 투시 변환 -> 적응형 이진화
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import order_points, target_size
from .preprocess import preprocess_for_edges


@dataclass
class PipelineResult:
    """파이프라인 실행 결과(단계별 이미지 + 검출 성공 여부)."""

    original: np.ndarray
    gray: np.ndarray
    blurred: np.ndarray
    edged: np.ndarray
    contour_vis: np.ndarray            # 검출된 사각형을 그린 원본
    warped: np.ndarray | None          # 투시 변환 결과(실패 시 None)
    scanned: np.ndarray | None         # 적응형 이진화 결과(실패 시 None)
    document_found: bool
    quad: np.ndarray | None            # 검출된 4점 좌표(정렬 전)


class DocumentScanner:
    """문서 검출 및 투시 변환 스캐너.

    Parameters
    ----------
    blur_kernel : int
        가우시안 블러 커널 크기(홀수).
    canny_low, canny_high : int
        Canny 임계값.
    min_area_ratio : float
        전체 이미지 대비 문서로 인정할 최소 면적 비율(노이즈 윤곽선 제거).
    approx_eps_ratio : float
        approxPolyDP 근사 정밀도(둘레 대비 비율).
    """

    def __init__(
        self,
        blur_kernel: int = 5,
        canny_low: int = 50,
        canny_high: int = 150,
        min_area_ratio: float = 0.08,
        approx_eps_ratio: float = 0.02,
    ) -> None:
        self.blur_kernel = blur_kernel
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.min_area_ratio = min_area_ratio
        self.approx_eps_ratio = approx_eps_ratio

    # ------------------------------------------------------------------ #
    # 1) 문서 윤곽선(사각형) 추출
    # ------------------------------------------------------------------ #
    def find_document_quad(self, edged: np.ndarray) -> np.ndarray | None:
        """엣지 이미지에서 꼭짓점 4개짜리 최대 사각형을 찾는다.

        - findContours로 닫힌 윤곽선 추출
        - 면적 필터링으로 작은 노이즈 제거
        - approxPolyDP로 꼭짓점 수가 4개인 볼록 다각형 선택
        """
        contours, _ = cv2.findContours(
            edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            return None

        img_area = edged.shape[0] * edged.shape[1]
        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area_ratio * img_area:
                break  # 정렬되어 있으므로 이보다 작은 것은 볼 필요 없음
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, self.approx_eps_ratio * peri, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx.reshape(4, 2).astype("float32")
        return None

    # ------------------------------------------------------------------ #
    # 2) 투시 변환
    # ------------------------------------------------------------------ #
    @staticmethod
    def four_point_transform(image: np.ndarray, quad: np.ndarray) -> np.ndarray:
        """4점을 정면 직사각형으로 펴는 투시 변환."""
        ordered = order_points(quad)
        width, height = target_size(ordered)

        dst = np.array(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
            dtype="float32",
        )
        matrix = cv2.getPerspectiveTransform(ordered, dst)
        return cv2.warpPerspective(image, matrix, (width, height))

    # ------------------------------------------------------------------ #
    # 3) 후처리(적응형 이진화)
    # ------------------------------------------------------------------ #
    @staticmethod
    def postprocess(warped: np.ndarray, block_size: int = 15, c: int = 10) -> np.ndarray:
        """그림자/조명 얼룩에 강인한 적응형 이진화로 스캔본을 만든다."""
        gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        block_size = block_size if block_size % 2 == 1 else block_size + 1
        return cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            block_size,
            c,
        )

    # ------------------------------------------------------------------ #
    # 전체 파이프라인
    # ------------------------------------------------------------------ #
    def run(self, image: np.ndarray) -> PipelineResult:
        """원본 이미지 한 장에 대해 전체 파이프라인을 실행한다."""
        steps = preprocess_for_edges(
            image, self.blur_kernel, self.canny_low, self.canny_high
        )
        quad = self.find_document_quad(steps["edged"])

        contour_vis = image.copy()
        warped = scanned = None
        found = quad is not None

        if found:
            cv2.drawContours(
                contour_vis, [quad.astype(int)], -1, (0, 255, 0), 3
            )
            for (x, y) in quad.astype(int):
                cv2.circle(contour_vis, (int(x), int(y)), 8, (0, 0, 255), -1)
            warped = self.four_point_transform(image, quad)
            scanned = self.postprocess(warped)
        else:
            cv2.putText(
                contour_vis,
                "No document detected",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2,
            )

        return PipelineResult(
            original=image,
            gray=steps["gray"],
            blurred=steps["blurred"],
            edged=steps["edged"],
            contour_vis=contour_vis,
            warped=warped,
            scanned=scanned,
            document_found=found,
            quad=quad,
        )


# --------------------------------------------------------------------- #
# 유틸
# --------------------------------------------------------------------- #
def read_image(path: str, max_side: int = 1200) -> np.ndarray:
    """이미지를 읽고, 과도하게 크면 긴 변 기준으로 축소한다(처리 속도 안정화)."""
    # np.fromfile + imdecode: 한글/유니코드 경로 지원
    data = np.fromfile(path, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")

    h, w = image.shape[:2]
    longest = max(h, w)
    if longest > max_side:
        scale = max_side / longest
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def imwrite_unicode(path: str, image: np.ndarray) -> bool:
    """유니코드 경로 지원 저장."""
    ext = "." + path.rsplit(".", 1)[-1] if "." in path else ".png"
    ok, buf = cv2.imencode(ext, image)
    if ok:
        buf.tofile(path)
    return bool(ok)
