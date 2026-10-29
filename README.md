# 📄 Document Scanner (A3-1)

휴대폰으로 비스듬히 찍은 종이를 자동으로 인식하여 **정면에서 본 것처럼 반듯하게 펴주는** 문서 스캐너.
OpenCV + NumPy로 `노이즈 제거 → 엣지 검출 → 윤곽선 추출 → 투시 변환 → 적응형 이진화`에 이르는
이미지 처리 파이프라인 전체를 직접 설계했다.

> 핵심 학습 목표: "사람이 규칙을 설계하는 방식"의 한계를 직접 체감하고, 왜 "데이터에서 학습하는 방식"이 필요한지 스스로 답하기.

---

## 1. 설치

```bash
pip install -r requirements.txt
```

- 필수: `opencv-python`, `numpy`, `matplotlib`
- 선택(보너스 3 PDF 내보내기 시에만): `Pillow`

---

## 2. 폴더 구조

```
document-scanner/
├── main.py                 # 대화형 앱 (트랙바 + 단계별 시각화)
├── webcam.py               # (선택) 웹캠 보드
├── requirements.txt
├── README.md
├── src/
│   ├── geometry.py         # 4점 좌표 정렬 (라이브러리 無, 기하 원리 직접 구현)
│   ├── preprocess.py       # Grayscale / Gaussian Blur / Canny
│   ├── scanner.py          # 문서 검출 + 투시 변환 + 후처리 파이프라인
│   ├── eda.py              # 픽셀 히스토그램 기반 EDA
│   ├── evaluate.py         # 20장 배치 성능 평가 (조건별 성공률 표 & 파라미터 딜레마 분석)
│   ├── convolution.py      # (보너스1) NumPy-only Gaussian Blur + 속도 비교
│   └── export_pdf.py       # (보너스3) 스캔본 -> PDF
├── tests/
│   └── test_pipeline.py    # 합성 이미지로 핵심 로직 헤드리스 검증
├── images/                 # 본인 촬영 사진 20장 + meta.csv (직접 준비)
└── report/
    └── analysis_report.md  # 분석 보고서 (템플릿 제공)
```

---

## 3. 사용법

### 3-1. 대화형 스캐너 (단계별 시각화 + 트랙바)

```bash
python main.py --image images/easy_01.jpg
python main.py --image images/hard_shadow_03.jpg --save output/result.png
```

- 창 1 `Pipeline`: 원본 | Grayscale | Blur | Edge
- 창 2 `Detection & Scan`: 윤곽선 검출 | 투시 변환 | 스캔본
- 창 3 `Controls`: **Blur / Canny Low / Canny High** 트랙바 실시간 조절
- 키: `s` 스캔 결과 저장, `q` 종료

### 3-2. 탐색적 분석 (EDA, 밝기 히스토그램)

```bash
python -m src.eda --dir images
```
파일명에 `easy`가 포함되면 쉬운 케이스로 분류하여 히스토그램을 비교 저장한다.

### 3-3. 배치 성능 평가 (20장 → 조건별 성공률 표 & 파라미터 딜레마 분석)

```bash
# 기본(단일) 설정 평가
python -m src.evaluate --dir images --meta images/meta.csv --out report

# 여러 파라미터 프리셋 비교 (파라미터 딜레마 정량 증명)
python -m src.evaluate --dir images --meta images/meta.csv --out report --compare
```

- **기본 평가 실행 시**:
  - `report/evaluation_summary.md`: 조건별(조명/각도/배경) 성공률 표
  - `report/evaluation_results.csv`: 파일별 자동 판정 결과 + `manual_success` 수기 검토 기입 열
  - `report/warped/`: 변환된 스캔본 이미지들(육안 검증용)
- **`--compare` 옵션 사용 시**:
  - `soft` (3/30/90), `default` (5/50/150), `aggressive` (7/75/200) 3가지 파라미터 프리셋을 20장에 일괄 적용
  - `report/param_comparison.md`: 프리셋별 전체 및 조건별 성공률 교차표, 파라미터 딜레마 증거(설정에 따라 성공/실패가 갈리는 케이스 목록) 자동 생성
  - `report/param_comparison.csv`: 파일별 프리셋 결과 원시 데이터
  - *→ "모든 이미지에 통하는 만능 고정 파라미터는 존재하지 않는다"는 규칙 기반의 한계를 수치로 입증하여 `analysis_report.md` 3절에 활용*

### 3-4. 웹캠 보드 (선택)

```bash
python webcam.py            # 기본 카메라
python webcam.py --video sample.mp4
```

### 3-5. 헤드리스 검증 (카메라/사진 없이 동작 확인)

```bash
python -m tests.test_pipeline
```
합성 문서 이미지를 생성해 좌표 정렬·검출·변환이 정상 동작하는지 확인한다.

### 보너스

```bash
# (보너스1) NumPy-only Gaussian Blur: 이중 for문 vs 벡터화 속도 비교
python -m src.convolution --image images/easy_01.jpg --size 5

# (보너스3) 스캔본 PDF 내보내기
python -m src.export_pdf --images output/a_scan.png output/b_scan.png --out output/scan.pdf
```

---

## 4. 파이프라인 설계

| 단계 | 모듈 | 핵심 함수 | 설명 |
|---|---|---|---|
| 전처리 | `preprocess.py` | `to_grayscale`, `gaussian_blur` | 색 정보 제거 + 노이즈 평활화 |
| 엣지 검출 | `preprocess.py` | `canny_edge` | 밝기 급변 지점을 외곽선 후보로 |
| 윤곽선 추출 | `scanner.py` | `find_document_quad` | 면적 필터 + `approxPolyDP`로 꼭짓점 4개 사각형 |
| 좌표 정렬 | `geometry.py` | `order_points` | 합(x+y)·차(y−x)로 TL·TR·BR·BL 결정 (직접 구현) |
| 투시 변환 | `scanner.py` | `four_point_transform` | `getPerspectiveTransform` + `warpPerspective` |
| 후처리 | `scanner.py` | `postprocess` | 적응형 이진화로 그림자에 강인한 스캔본 |

**좌표 정렬 원리** (라이브러리 비의존):
이미지 좌표계에서 좌상단은 `x+y`가 최소, 우하단은 `x+y`가 최대,
우상단은 `y−x`가 최소, 좌하단은 `y−x`가 최대라는 기하학적 성질만으로 네 점의 순서를 결정한다.

**종횡비 자동 보정(보너스2)**: 변환 목표 크기를 고정하지 않고, 검출된 네 변의 실제 픽셀 길이를 측정해
결과물이 찌그러지지 않도록 가로/세로를 계산한다(`geometry.target_size`).

---

## 5. 과제 목표 자가 점검

- [ ] 이미지가 (H, W, C) 3차원 배열임을 이해하고 NumPy로 조작했는가
- [ ] 컨볼루션이 필터와 이미지의 내적임을 이해했는가 (`convolution.py`)
- [ ] 행렬 곱으로 기하학적 변환(투시)을 수행하는 원리를 설명할 수 있는가
- [ ] 고정 파라미터가 다양한 환경에서 왜 실패하는지 정량 데이터로 설명했는가 (`report/`)
- [ ] 규칙 기반의 한계 → 학습 기반 필요성을 자기 경험으로 서술했는가 (`report/analysis_report.md`)

---

## 6. 분석 보고서

`report/analysis_report.md` 참고. 다음을 포함한다.
1. 테스트 결과 요약(조건별 성공률 표)
2. 실패 원인 분석(실패 케이스별)
3. 파라미터 딜레마(모든 이미지 공통 최적 파라미터는 존재하는가 → `--compare`로 정량 증명)
4. 규칙 기반의 한계
5. 학습 기반 접근이 필요한 이유
