"""문서 스캐너 대화형 애플리케이션.

사용 예:
    python main.py --image images/receipt.jpg
    python main.py --image images/receipt.jpg --save output/receipt_scan.png

기능:
    - 처리 과정(원본 -> 전처리 -> 엣지 -> 윤곽선 -> 변환 -> 스캔)을 단계별 윈도우로 시각화
    - 트랙바로 블러 강도 / Canny 임계값(low, high)을 실시간 조절
    - 's' 키: 스캔 결과 저장, 'q' 키: 종료
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np

from src.scanner import DocumentScanner, imwrite_unicode, read_image

WINDOW_CONTROL = "Controls"
WINDOW_STEPS = "Pipeline (original | gray | blur | edge)"
WINDOW_RESULT = "Detection & Scan (contour | warped | scanned)"


def _to_bgr(img: np.ndarray) -> np.ndarray:
    """1채널 이미지를 3채널로 맞춰 가로로 이어붙일 수 있게 한다."""
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _resize_h(img: np.ndarray, height: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = height / h
    return cv2.resize(img, (max(int(w * scale), 1), height))


def hstack_panels(images: list[np.ndarray], height: int = 320) -> np.ndarray:
    """서로 크기가 다른 이미지를 같은 높이로 맞춰 가로로 이어붙인다."""
    panels = [_resize_h(_to_bgr(im), height) for im in images if im is not None]
    if not panels:
        return np.zeros((height, height, 3), dtype=np.uint8)
    return cv2.hconcat(panels)


def nothing(_x: int) -> None:
    pass


def build_controls() -> None:
    cv2.namedWindow(WINDOW_CONTROL, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_CONTROL, 420, 140)
    # 블러 커널(0~15, 내부에서 홀수 보정)
    cv2.createTrackbar("Blur", WINDOW_CONTROL, 5, 15, nothing)
    cv2.createTrackbar("Canny Low", WINDOW_CONTROL, 50, 255, nothing)
    cv2.createTrackbar("Canny High", WINDOW_CONTROL, 150, 255, nothing)


def read_controls(scanner: DocumentScanner) -> None:
    scanner.blur_kernel = cv2.getTrackbarPos("Blur", WINDOW_CONTROL)
    scanner.canny_low = cv2.getTrackbarPos("Canny Low", WINDOW_CONTROL)
    scanner.canny_high = cv2.getTrackbarPos("Canny High", WINDOW_CONTROL)


def main() -> None:
    parser = argparse.ArgumentParser(description="문서 스캐너 (대화형)")
    parser.add_argument("--image", required=True, help="입력 이미지 경로")
    parser.add_argument("--save", default=None, help="스캔 결과 저장 경로")
    args = parser.parse_args()

    image = read_image(args.image)
    scanner = DocumentScanner()

    build_controls()
    cv2.namedWindow(WINDOW_STEPS, cv2.WINDOW_NORMAL)
    cv2.namedWindow(WINDOW_RESULT, cv2.WINDOW_NORMAL)

    print("[사용법] 트랙바로 파라미터 조절 | 's' 저장 | 'q' 종료")

    while True:
        read_controls(scanner)
        result = scanner.run(image)

        steps_panel = hstack_panels(
            [result.original, result.gray, result.blurred, result.edged]
        )
        blank = np.zeros_like(result.original)
        result_panel = hstack_panels(
            [
                result.contour_vis,
                result.warped if result.warped is not None else blank,
                result.scanned if result.scanned is not None else blank,
            ]
        )

        status = "FOUND" if result.document_found else "NOT FOUND"
        cv2.putText(
            result_panel,
            f"Document: {status}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0) if result.document_found else (0, 0, 255),
            2,
        )

        cv2.imshow(WINDOW_STEPS, steps_panel)
        cv2.imshow(WINDOW_RESULT, result_panel)

        key = cv2.waitKey(30) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s"):
            if result.scanned is not None:
                save_path = args.save or _default_save_path(args.image)
                os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
                imwrite_unicode(save_path, result.scanned)
                print(f"[저장됨] {save_path}")
            else:
                print("[경고] 문서를 검출하지 못해 저장할 수 없습니다. 파라미터를 조절하세요.")

    cv2.destroyAllWindows()


def _default_save_path(image_path: str) -> str:
    base = os.path.splitext(os.path.basename(image_path))[0]
    return os.path.join("output", f"{base}_scan.png")


if __name__ == "__main__":
    main()
