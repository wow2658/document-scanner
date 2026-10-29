# 📄 A3-1 Document Scanner 프로젝트 이주 및 개발 내역 요약

이 문서는 이전 세션(112턴 대화) 및 작업 공간(`A4-2 글속좋음나쁨자동`)에서 논의되고 구현된 전체 내용을 `A3-1 document-scanner` 워크스페이스로 체계적으로 정리·이주한 문서입니다.

---

## 1. 프로젝트 개요 및 선정 배경

* **미션명**: A3-1 Document Scanner
* **과제 정의**: 휴대폰으로 비스듬히 촬영한 종이를 자동으로 감지하여 **정면에서 스캔한 것처럼 반듯하게 펴주는** 규칙 기반 문서 스캐너 파이프라인 구축.
* **핵심 학습 목표**:
  1. 이미지가 `(H, W, C)` 3차원 배열임을 이해하고 NumPy/OpenCV로 조작.
  2. 컨볼루션(2D 커널 필터와 이미지 영역의 내적)을 통한 노이즈 제거 원리 체득.
  3. 기하학적 2D 투시 변환(Perspective Transform)의 3x3 변환 행렬 도출 및 워핑 원리 이해.
  4. **파라미터 딜레마(Parameter Dilemma)**: 조명·각도·배경이 다양한 실제 환경에서 고정 규칙 및 단일 파라미터가 왜 실패하는지 정량 데이터로 증명.
  5. 규칙 기반(Rule-based) 접근의 근본적 한계를 파악하고, 왜 **데이터 기반 딥러닝/머신러닝(Learning-based)** 접근이 필요한지 스스로 설명하기.

---

## 2. 전체 파일 구조 및 모듈별 역할

```
A3-1 document-scanner/
├── main.py                 # 대화형 데모 앱 (트랙바를 통한 실시간 파라미터 튜닝 및 3단계 시각화)
├── webcam.py               # (선택) 웹캠 또는 동영상 파일 실시간 프레임 스캐너
├── requirements.txt        # 의존성 목록 (opencv-python, numpy, matplotlib, Pillow)
├── README.md               # 프로젝트 실행 가이드 및 명령어 매뉴얼
├── MIGRATION_SUMMARY.md    # [본 문서] 이전 작업 세션 전체 이주 및 히스토리 정리
├── src/
│   ├── geometry.py         # 4점 좌표 정렬 (외부 라이브러리 없이 순수 기하 원리로 TL/TR/BR/BL 정렬) + 종횡비 자동 보정
│   ├── preprocess.py       # Grayscale 변환, Gaussian Blur, Canny Edge Detection
│   ├── scanner.py          # 윤곽선 검출(find_document_quad) + 투시 변환 + 적응형 이진화 후처리
│   ├── eda.py              # 데이터 밝기 히스토그램 비교 EDA 도구
│   ├── evaluate.py         # 20장 배치 성능 평가 (--compare를 통한 파라미터 딜레마 분석 지원)
│   ├── convolution.py      # (보너스 1) 순수 NumPy 구현 2D Gaussian Blur (이중 for문 vs 벡터화 벤치마크)
│   └── export_pdf.py       # (보너스 3) 처리된 스캔 결과 이미지를 단일 PDF로 묶어 내보내기
├── tests/
│   └── test_pipeline.py    # 합성 문서 이미지를 생성하여 헤드리스(UI 없이) 파이프라인 검증
├── images/
│   └── meta.csv            # 20장 사진 메타데이터 (파일명, 조건, 조명, 각도, 배경 복잡도)
└── report/
    └── analysis_report.md  # 최종 과제 분석 보고서 작성 템플릿
```

---

## 3. 핵심 설계 및 구현 상세

### 3-1. 기하학적 좌표 정렬 (`src/geometry.py`)
* 외부 편의 함수(예: imutils)에 의존하지 않고 좌표계 성질만으로 구현:
  * $x + y$가 최소인 점 $\rightarrow$ **좌상단 (Top-Left)**
  * $x + y$가 최대인 점 $\rightarrow$ **우하단 (Bottom-Right)**
  * $y - x$가 최소인 점 $\rightarrow$ **우상단 (Top-Right)**
  * $y - x$가 최대인 점 $\rightarrow$ **좌하단 (Bottom-Left)**
* **종횡비 자동 보정 (`target_size`)**:
  * 사각형의 상/하 변 길이 평균(너비)과 좌/우 변 길이 평균(높이)을 유클리드 거리로 계산하여 출력 이미지 종횡비가 원본 문서 비율에 맞게 왜곡 없이 복원되도록 설계.

### 3-2. 전처리 및 스캐너 파이프라인 (`src/preprocess.py`, `src/scanner.py`)
1. **Grayscale**: 불필요한 색상 정보 제거 및 연산량 절감.
2. **Gaussian Blur**: 고주파 노이즈 및 자잘한 텍스처 평활화.
3. **Canny Edge**: 그래디언트 크기와 방향을 기반으로 강한 엣지와 약한 엣지를 연결(이중 임계값).
4. **윤곽선 추출 (`cv2.findContours`)**:
   * 외곽선 면적 내림차순 정렬 후, `cv2.arcLength` 및 `cv2.approxPolyDP`를 통해 꼭짓점이 4개인 볼록 다각형 탐색.
5. **투시 변환 (`cv2.getPerspectiveTransform`, `cv2.warpPerspective`)**:
   * 원본 4점과 목표 직사각형 4점 간의 3x3 호모그래피 행렬 계산 및 보간.
6. **적응형 이진화 (`postprocess`)**:
   * 국소 영역별 가중 평균을 기준으로 임계값을 계산하는 `cv2.ADAPTIVE_THRESH_GAUSSIAN_C`를 적용하여 그림자나 불균일 조명 하에서도 텍스트를 선명하게 분리.

### 3-3. 다중 프리셋 비교 및 파라미터 딜레마 분석 (`src/evaluate.py --compare`)
* **3가지 프리셋 정의**:
  * `soft`: Blur=3, Canny=(30, 90) — 저대비·어두운 환경에 유리하나 노이즈에 취약
  * `default`: Blur=5, Canny=(50, 150) — 균형 잡힌 기본값
  * `aggressive`: Blur=7, Canny=(75, 200) — 강한 노이즈 억제하나 약한 경계선 누락
* **정량 증명**:
  * 20개 테스트 이미지에 대해 각 프리셋을 실행하여 `report/param_comparison.md`에 교차표 및 설정에 따라 성공/실패가 갈리는 이미지 목록을 자동 기록.
  * "A 이미지에 최적인 설정이 B 이미지에서는 실패한다"는 딜레마를 수치로 증명하여 보고서 3절의 강력한 근거로 활용.

---

## 4. 향후 남은 작업 가이드 (To-Do)

1. **실제 사진 20장 준비 (`images/`)**:
   * 휴대폰 카메라로 아래 조건에 맞추어 각 5장씩 총 20장 촬영 후 `images/` 폴더에 배치:
     * `easy_01.jpg` ~ `easy_05.jpg`: 단순 배경, 균일 조명, 정면 촬영
     * `hard_shadow_01.jpg` ~ `hard_shadow_05.jpg`: 스탠드 조명, 손 그림자 등 음영 발생
     * `hard_tilt_01.jpg` ~ `hard_tilt_05.jpg`: 45도 이상 비스듬한 각도 촬영
     * `hard_complex_01.jpg` ~ `hard_complex_05.jpg`: 나무 무늬 책상, 잡지 위, 키보드 주변 등 복잡한 배경
   * `images/meta.csv` 파일에 조건과 특이사항 기재.

2. **배치 평가 및 파라미터 비교 실행**:
   ```bash
   python -m src.evaluate --dir images --meta images/meta.csv --out report --compare
   ```
   * `report/evaluation_summary.md`, `report/param_comparison.md`, `report/warped/` 생성 확인.

3. **최종 분석 보고서 작성 (`report/analysis_report.md`)**:
   * 1절: 테스트 요약 수치 기입
   * 2절: 실패 사례 최소 3건 원인 분석 (예: 복잡한 배경으로 인한 외곽 사각형 오인식 등)
   * 3절: 파라미터 딜레마 (프리셋별 결과 비교 데이터 인용)
   * 4절 & 5절: 규칙 기반의 한계와 딥러닝/학습 기반 필요성 서술.

