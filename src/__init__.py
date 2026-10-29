"""문서 스캐너 패키지.

모듈 구성
---------
- geometry   : 4개 꼭짓점 좌표 정렬(직접 구현)
- preprocess : Grayscale, Gaussian Blur, Canny 엣지 검출
- scanner    : 문서 검출 + 투시 변환 + 후처리 파이프라인
- eda        : 픽셀 히스토그램 기반 탐색적 분석
- evaluate   : 20장 배치 성능 평가
- convolution: (보너스) NumPy만으로 구현한 Gaussian Blur
"""

from .scanner import DocumentScanner, PipelineResult

__all__ = ["DocumentScanner", "PipelineResult"]
