"""(보너스 3) 처리된 이미지를 PDF로 내보내기.

보너스 3 한정으로 Pillow(PIL) 사용이 허용된다.

사용법:
    python -m src.export_pdf --images output/a_scan.png output/b_scan.png --out output/scan.pdf
"""

from __future__ import annotations

import argparse

from PIL import Image


def images_to_pdf(image_paths: list[str], out_path: str) -> str:
    """여러 이미지를 한 개의 PDF로 합친다(각 이미지가 한 페이지)."""
    if not image_paths:
        raise ValueError("이미지 경로가 비어 있습니다.")

    pages = [Image.open(p).convert("RGB") for p in image_paths]
    first, rest = pages[0], pages[1:]
    first.save(out_path, "PDF", resolution=200.0, save_all=True, append_images=rest)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="스캔 이미지 -> PDF")
    ap.add_argument("--images", nargs="+", required=True, help="PDF로 묶을 이미지들")
    ap.add_argument("--out", default="output/scan.pdf")
    args = ap.parse_args()
    saved = images_to_pdf(args.images, args.out)
    print(f"[저장됨] {saved}")
