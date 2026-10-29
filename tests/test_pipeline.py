"""헤드리스 파이프라인 검증 스크립트(합성 이미지 사용).

GUI 없이 핵심 로직(좌표 정렬 -> 사각형 검출 -> 투시 변환 -> 후처리)이
정상 동작하는지 확인한다. 실제 제출용 평가는 본인 사진 20장으로 수행한다.

실행:
    python -m tests.test_pipeline
"""

from __future__ import annotations

import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.geometry import order_points, target_size  # noqa: E402
from src.scanner import DocumentScanner, imwrite_unicode  # noqa: E402


def make_synthetic_document(
    canvas_size=(800, 600), tilt=0.25, seed=0
) -> tuple[np.ndarray, np.ndarray]:
    """복잡한 배경 위에 기울어진 '문서'가 놓인 합성 이미지를 만든다."""
    rng = np.random.default_rng(seed)
    h, w = canvas_size
    # 노이즈 배경
    bg = rng.integers(40, 120, size=(h, w, 3), dtype=np.uint8)
    for _ in range(40):
        x1, y1 = rng.integers(0, w), rng.integers(0, h)
        x2, y2 = rng.integers(0, w), rng.integers(0, h)
        color = tuple(int(c) for c in rng.integers(0, 150, size=3))
        cv2.line(bg, (x1, y1), (x2, y2), color, 2)

    # 밝은 흰 종이(사각형) + 투시 왜곡
    pw, ph = 360, 480
    paper = np.full((ph, pw, 3), 245, np.uint8)
    for i in range(6):
        cv2.putText(
            paper, "Sample Document Line", (20, 60 + i * 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 30, 30), 2,
        )

    src = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype="float32")
    cx, cy = w // 2, h // 2
    dst = np.array(
        [
            [cx - 170 + tilt * 60, cy - 230],
            [cx + 190, cy - 200 + tilt * 40],
            [cx + 150 - tilt * 50, cy + 240],
            [cx - 200, cy + 210],
        ],
        dtype="float32",
    )
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(paper, M, (w, h), borderValue=(0, 0, 0))
    mask = cv2.warpPerspective(
        np.full((ph, pw), 255, np.uint8), M, (w, h)
    )
    result = bg.copy()
    result[mask > 0] = warped[mask > 0]
    return result, dst


def test_order_points() -> None:
    """정렬이 TL, TR, BR, BL 순서를 보장하는지 확인."""
    pts = np.array([[100, 100], [10, 20], [120, 15], [5, 130]], dtype="float32")
    o = order_points(pts)
    assert np.allclose(o[0], [10, 20]), f"TL 오류: {o[0]}"
    assert np.allclose(o[1], [120, 15]), f"TR 오류: {o[1]}"
    assert np.allclose(o[2], [100, 100]), f"BR 오류: {o[2]}"
    assert np.allclose(o[3], [5, 130]), f"BL 오류: {o[3]}"
    w, h = target_size(o)
    assert w > 0 and h > 0
    print("[OK] order_points / target_size")


def test_full_pipeline() -> None:
    """합성 문서에서 4점 검출 + 투시 변환이 성공하는지 확인."""
    os.makedirs("output", exist_ok=True)
    scanner = DocumentScanner()

    success = 0
    for seed in range(5):
        img, _ = make_synthetic_document(seed=seed, tilt=0.1 * seed)
        result = scanner.run(img)
        if result.document_found and result.scanned is not None:
            success += 1
            if seed == 0:
                imwrite_unicode("output/_test_original.png", img)
                imwrite_unicode("output/_test_edged.png", result.edged)
                imwrite_unicode("output/_test_contour.png", result.contour_vis)
                imwrite_unicode("output/_test_warped.png", result.warped)
                imwrite_unicode("output/_test_scanned.png", result.scanned)

    print(f"[OK] full_pipeline: 합성 문서 5장 중 {success}장 검출 성공")
    assert success >= 4, "합성 문서 검출률이 낮습니다. 파라미터를 점검하세요."


if __name__ == "__main__":
    test_order_points()
    test_full_pipeline()
    print("\n모든 테스트 통과. output/_test_*.png 에서 단계별 결과를 확인하세요.")
