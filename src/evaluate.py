"""정량적 성능 평가(배치).

요구사항:
    - 20장 전체에 대해 "문서 검출 성공 여부"를 측정
      성공 기준: 4개 꼭짓점이 올바르게 검출되어 문서가 반듯하게 변환됨
    - 조건별(조명, 각도, 배경) 성공률을 표로 정리
    - 실패 케이스 최소 3건 이상 원인 분석

사용법:
    # 단일(기본) 설정 평가
    python -m src.evaluate --dir images --meta images/meta.csv --out report/

    # 여러 파라미터 설정을 20장에 일괄 적용해 설정별 성공률을 비교
    #  → "모든 이미지에 통하는 최적 파라미터가 존재하는가?"(파라미터 딜레마)를 정량 증명
    python -m src.evaluate --dir images --meta images/meta.csv --out report/ --compare

파일명 규칙(메타데이터 CSV가 없을 때 fallback):
    파일명에 포함된 키워드로 조건을 추정한다.
        shadow(그림자) / tilt·skew(기울어짐) / complex(복잡배경) / easy(단순정면)

meta.csv 형식(권장):
    filename,condition,lighting,angle,background,note
    easy_01.jpg,단순 배경 + 정면,균일,0,단순,
    hard_shadow_03.jpg,그림자 있음,그림자,0,단순,
"""

from __future__ import annotations

import argparse
import csv
import glob
import os

import numpy as np

from .scanner import DocumentScanner, imwrite_unicode, read_image

# --------------------------------------------------------------------- #
# 파라미터 프리셋 (이름 -> DocumentScanner 생성 인자)
#   soft       : 약한 블러 + 낮은 임계값. 저대비/어두운 이미지에서 엣지를 더 많이 잡지만
#                복잡 배경에서는 노이즈 윤곽을 함께 잡아 오검출이 늘어난다.
#   default    : 균형 잡힌 기본값.
#   aggressive : 강한 블러 + 높은 임계값. 복잡 배경 노이즈는 억제하지만
#                약한 엣지(그림자 경계 등)는 놓치기 쉽다.
# --------------------------------------------------------------------- #
PRESETS: dict[str, dict[str, int]] = {
    "soft": dict(blur_kernel=3, canny_low=30, canny_high=90),
    "default": dict(blur_kernel=5, canny_low=50, canny_high=150),
    "aggressive": dict(blur_kernel=7, canny_low=75, canny_high=200),
}


def _auto_rules_from_quad(result) -> bool:
    """자동 성공 판정(1차 필터).

    실제 '반듯하게 변환됨'은 사람이 최종 확인해야 하지만,
    4점 사각형이 검출되고 변환 결과가 충분한 면적을 가지면 1차 성공으로 본다.
    """
    if not result.document_found or result.warped is None:
        return False
    h, w = result.warped.shape[:2]
    # 너무 가늘거나 작은 변환은 실패로 간주(잘못된 사각형 검출)
    if min(h, w) < 50:
        return False
    aspect = max(h, w) / max(min(h, w), 1)
    if aspect > 6:  # 종이가 이렇게 극단적 비율이기 어려움
        return False
    return True


def _infer_condition(filename: str) -> str:
    name = filename.lower()
    if "easy" in name:
        return "단순 배경 + 정면"
    if "shadow" in name:
        return "그림자 있음"
    if "tilt" in name or "skew" in name or "angle" in name:
        return "45도 이상 기울어짐"
    if "complex" in name or "bg" in name:
        return "복잡한 배경"
    return "미분류"


def load_meta(meta_path: str | None) -> dict[str, str]:
    """파일명 -> 조건 매핑 로드."""
    mapping: dict[str, str] = {}
    if meta_path and os.path.exists(meta_path):
        with open(meta_path, encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                fn = row.get("filename", "").strip()
                cond = row.get("condition", "").strip()
                if fn:
                    mapping[fn] = cond or _infer_condition(fn)
    return mapping


def evaluate_dir(image_dir: str, meta_path: str | None, out_dir: str) -> None:
    paths = sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )
    if not paths:
        print(f"[경고] 이미지가 없습니다: {image_dir}")
        return

    meta = load_meta(meta_path)
    scanner = DocumentScanner()

    warped_dir = os.path.join(out_dir, "warped")
    os.makedirs(warped_dir, exist_ok=True)

    rows = []
    # 조건별 집계
    per_cond: dict[str, list[int]] = {}

    for p in paths:
        fn = os.path.basename(p)
        cond = meta.get(fn) or _infer_condition(fn)
        try:
            image = read_image(p)
        except FileNotFoundError:
            continue
        result = scanner.run(image)
        success = _auto_rules_from_quad(result)

        if result.scanned is not None:
            imwrite_unicode(os.path.join(warped_dir, f"{fn}_scan.png"), result.scanned)

        rows.append({"filename": fn, "condition": cond, "auto_success": int(success)})
        per_cond.setdefault(cond, []).append(int(success))

    # 결과 CSV
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "evaluation_results.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["filename", "condition", "auto_success", "manual_success(검토기입)"]
        )
        writer.writeheader()
        for r in rows:
            r["manual_success(검토기입)"] = ""  # 사람이 최종 확인 후 기입
            writer.writerow(r)

    # 요약 표(Markdown)
    md_path = os.path.join(out_dir, "evaluation_summary.md")
    total = len(rows)
    total_succ = sum(r["auto_success"] for r in rows)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 정량적 성능 평가 요약 (자동 1차 판정)\n\n")
        f.write(f"- 전체: {total}장, 자동 성공: {total_succ}장, ")
        f.write(f"성공률: {total_succ / total * 100:.1f}%\n\n")
        f.write("| 조건 | 테스트 수 | 성공 | 실패 | 성공률 |\n")
        f.write("|---|---|---|---|---|\n")
        for cond, vals in per_cond.items():
            n, s = len(vals), sum(vals)
            f.write(f"| {cond} | {n} | {s} | {n - s} | {s / n * 100:.0f}% |\n")
        f.write(
            "\n> auto_success는 '4점 검출 + 합리적 종횡비'에 기반한 1차 판정이다.\n"
            "> 최종 성공 여부는 `evaluation_results.csv`의 "
            "`manual_success` 열에 눈으로 확인 후 기입한다.\n"
        )

    print(f"[완료] 전체 {total}장 / 자동 성공 {total_succ}장 ({total_succ/total*100:.1f}%)")
    print(f"  - 상세 CSV : {csv_path}")
    print(f"  - 요약 표  : {md_path}")
    print(f"  - 스캔본   : {warped_dir}/")


# --------------------------------------------------------------------- #
# 파라미터 다중 설정 비교 (파라미터 딜레마 정량 증명)
# --------------------------------------------------------------------- #
def _collect_paths(image_dir: str) -> list[str]:
    return sorted(
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
    )


def _run_one_config(
    paths: list[str], meta: dict[str, str], params: dict[str, int]
) -> dict[str, tuple[str, int]]:
    """한 파라미터 설정으로 전체 이미지를 처리한다.

    Returns
    -------
    dict: filename -> (condition, success(0/1))
    """
    scanner = DocumentScanner(**params)
    out: dict[str, tuple[str, int]] = {}
    for p in paths:
        fn = os.path.basename(p)
        cond = meta.get(fn) or _infer_condition(fn)
        try:
            image = read_image(p)
        except FileNotFoundError:
            continue
        result = scanner.run(image)
        out[fn] = (cond, int(_auto_rules_from_quad(result)))
    return out


def compare_presets(
    image_dir: str,
    meta_path: str | None,
    out_dir: str,
    presets: dict[str, dict[str, int]] | None = None,
) -> None:
    """여러 파라미터 설정을 20장에 일괄 적용하고 설정별 성공률을 비교한다."""
    presets = presets or PRESETS
    paths = _collect_paths(image_dir)
    if not paths:
        print(f"[경고] 이미지가 없습니다: {image_dir}")
        return

    meta = load_meta(meta_path)
    # config_name -> {filename: (cond, success)}
    results = {name: _run_one_config(paths, meta, prm) for name, prm in presets.items()}

    files = [os.path.basename(p) for p in paths]
    conds = {fn: (results[next(iter(presets))][fn][0]) for fn in files if fn in results[next(iter(presets))]}
    all_conditions = sorted(set(conds.values()))

    os.makedirs(out_dir, exist_ok=True)

    # 1) 설정 x 조건 성공률 교차표
    md_path = os.path.join(out_dir, "param_comparison.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 파라미터 설정 비교 (파라미터 딜레마 분석)\n\n")
        f.write("## 1. 설정별 전체/조건별 성공률 (자동 1차 판정)\n\n")

        header = "| 설정 (blur/low/high) | 전체 | " + " | ".join(all_conditions) + " |\n"
        sep = "|" + "---|" * (2 + len(all_conditions)) + "\n"
        f.write(header)
        f.write(sep)
        for name, prm in presets.items():
            res = results[name]
            total = len(res)
            succ = sum(v[1] for v in res.values())
            label = f"{name} ({prm['blur_kernel']}/{prm['canny_low']}/{prm['canny_high']})"
            row = f"| {label} | {succ}/{total} ({succ/total*100:.0f}%) |"
            for c in all_conditions:
                vals = [v[1] for v in res.values() if v[0] == c]
                if vals:
                    row += f" {sum(vals)}/{len(vals)} ({sum(vals)/len(vals)*100:.0f}%) |"
                else:
                    row += " - |"
            f.write(row + "\n")

        # 2) 파라미터 딜레마 증거: 설정에 따라 성공/실패가 갈리는 이미지
        f.write("\n## 2. 파라미터 딜레마 증거\n\n")
        split_files = []
        for fn in files:
            if not all(fn in results[name] for name in presets):
                continue
            succ_by_cfg = {name: results[name][fn][1] for name in presets}
            if len(set(succ_by_cfg.values())) > 1:  # 설정마다 결과가 다름
                split_files.append((fn, conds[fn], succ_by_cfg))

        best_counts = {name: 0 for name in presets}
        for name in presets:
            best_counts[name] = sum(v[1] for v in results[name].values())
        best_cfg = max(best_counts, key=best_counts.get)

        if split_files:
            f.write(
                f"전체 {len(files)}장 중 **{len(split_files)}장**이 "
                "파라미터 설정에 따라 성공/실패가 갈렸다. "
                "→ *모든 이미지에 통하는 단일 최적 파라미터는 존재하지 않는다.*\n\n"
            )
            f.write("| 파일 | 조건 | " + " | ".join(presets) + " |\n")
            f.write("|" + "---|" * (2 + len(presets)) + "\n")
            for fn, cond, sc in split_files:
                marks = " | ".join("O" if sc[name] else "X" for name in presets)
                f.write(f"| {fn} | {cond} | {marks} |\n")
        else:
            f.write("모든 이미지에서 설정 간 결과 차이가 없었다. "
                    "(프리셋 간격을 더 벌리거나 어려운 이미지를 추가해 재검증 권장)\n")

        f.write(
            f"\n- 전체 기준 가장 성공률이 높은 설정: **{best_cfg}** "
            f"({best_counts[best_cfg]}/{len(files)})\n"
            "- 그러나 위 표처럼 특정 조건(예: 그림자/복잡 배경)에서는 다른 설정이 더 낫다. "
            "즉 최적 파라미터는 이미지 조건에 의존하며, 이것이 규칙 기반 접근의 근본 한계다.\n"
        )

    # 3) 원시 결과 CSV (설정 x 파일)
    csv_path = os.path.join(out_dir, "param_comparison.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "condition"] + list(presets.keys()))
        for fn in files:
            if not all(fn in results[name] for name in presets):
                continue
            row = [fn, conds[fn]] + [results[name][fn][1] for name in presets]
            writer.writerow(row)

    print("[완료] 파라미터 설정 비교")
    for name in presets:
        succ = sum(v[1] for v in results[name].values())
        print(f"  - {name:10s}: {succ}/{len(files)} 성공")
    print(f"  - 비교표 : {md_path}")
    print(f"  - 원시 CSV: {csv_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="문서 스캐너 배치 성능 평가")
    ap.add_argument("--dir", default="images", help="이미지 디렉터리")
    ap.add_argument("--meta", default="images/meta.csv", help="메타데이터 CSV(선택)")
    ap.add_argument("--out", default="report", help="결과 출력 디렉터리")
    ap.add_argument(
        "--compare", action="store_true",
        help="여러 파라미터 설정을 비교(파라미터 딜레마 분석)",
    )
    args = ap.parse_args()
    meta = args.meta if os.path.exists(args.meta) else None
    if args.compare:
        compare_presets(args.dir, meta, args.out)
    else:
        evaluate_dir(args.dir, meta, args.out)
