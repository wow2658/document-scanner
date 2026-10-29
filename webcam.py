"""(선택) 웹캠 보드.

- 웹캠 실시간 프레임에서 문서 검출 및 윤곽선 표시
- 's' : 현재 프레임 캡처 -> 변환 결과 저장
- 'q' : 종료

사용법:
    python webcam.py                # 기본 카메라(0)
    python webcam.py --cam 1        # 다른 카메라
    python webcam.py --video a.mp4  # 영상 파일 입력
"""

from __future__ import annotations

import argparse
import os
import time

import cv2

from src.scanner import DocumentScanner, imwrite_unicode


def main() -> None:
    ap = argparse.ArgumentParser(description="문서 스캐너 웹캠 보드")
    ap.add_argument("--cam", type=int, default=0, help="카메라 인덱스")
    ap.add_argument("--video", default=None, help="영상 파일 경로(지정 시 카메라 대신 사용)")
    ap.add_argument("--outdir", default="output", help="캡처 저장 폴더")
    args = ap.parse_args()

    source = args.video if args.video else args.cam
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[오류] 입력을 열 수 없습니다: {source}")
        return

    scanner = DocumentScanner()
    os.makedirs(args.outdir, exist_ok=True)
    saved = 0
    print("[사용법] 's' 캡처 저장 | 'q' 종료")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        result = scanner.run(frame)
        vis = result.contour_vis
        status = "FOUND" if result.document_found else "SEARCHING..."
        cv2.putText(
            vis, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0) if result.document_found else (0, 0, 255), 2,
        )
        cv2.imshow("Webcam - Document Detection", vis)
        if result.scanned is not None:
            cv2.imshow("Scanned Preview", result.scanned)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("s") and result.scanned is not None:
            path = os.path.join(args.outdir, f"capture_{int(time.time())}.png")
            imwrite_unicode(path, result.scanned)
            saved += 1
            print(f"[저장됨] {path}")

    cap.release()
    cv2.destroyAllWindows()
    print(f"총 {saved}장 저장됨.")


if __name__ == "__main__":
    main()
