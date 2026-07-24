# ShelfVisor 이미지 분석 패키지 설계

## 목적

현재 `app/analysis.py`에 집중된 이미지 준비, YOLO 후처리, OCR 후처리,
청구기호 판정, 결과 렌더링, 디버그 자료 생성을 책임별 모듈로 분리한다.

이번 변경의 목표는 다음 두 가지를 같은 수준으로 달성하는 것이다.

- 각 처리 책임을 독립적으로 교체하고 테스트할 수 있게 한다.
- 이미지 입력부터 최종 결과까지의 중간 상태를 단계별로 관찰하고 보존할 수 있게 한다.

구조 분리 과정에서는 기존 API 요청·응답 형식과 분석 동작을 우선 보존한다.
알고리즘 개선은 이 리팩터링의 범위에 포함하지 않는다.

## 패키지 구조

```text
app/
  analysis/
    __init__.py
    analysis_pipeline.py
    analysis_models.py
    image_processing.py

    yolo/
      __init__.py
      yolo_inference.py
      yolo_processing.py

    ocr/
      __init__.py
      ocr_inference.py
      ocr_processing.py

    call_number_processing.py
    result_rendering.py
    debug_artifacts.py

  server.py
```

기존 `app/analysis.py`는 `app/analysis/` 패키지로 대체한다. 패키지의
`__init__.py`는 `analyze_shelf_photo` 등 외부에서 사용하는 공개 진입점만
다시 노출한다. 따라서 `server.py`는 내부 파일 구조를 알 필요가 없다.

## 모듈 책임

### `analysis_pipeline.py`

전체 분석 순서만 조율한다.

1. 입력 이미지를 준비한다.
2. YOLO 추론과 후처리를 실행한다.
3. OCR 입력을 구성하고 OCR 추론과 후처리를 실행한다.
4. 청구기호 순서를 판정한다.
5. 최종 결과를 구성하고 이미지를 렌더링한다.
6. 요청된 경우 디버그 자료를 API 응답과 로컬 파일로 내보낸다.

좌표 계산, 이미지 렌더링, OCR 텍스트 선택 같은 세부 알고리즘은 포함하지
않는다.

### `analysis_models.py`

모듈 사이에서 전달되는 명시적인 데이터 구조를 정의한다. 최소 모델은
다음과 같다.

- `PreparedImage`: 분석에 사용할 Pillow 이미지와 원본 크기 정보
- `DetectedRegion`: box, polygon, confidence, class 정보
- `OCRContactSheet`: OCR용 이미지와 원본 영역 간 좌표 매핑
- `OCRRowResult`: 행별 토큰, 방향별 결과, 선택된 텍스트
- `SpineAnalysis`: 책등 영역, 인식 텍스트, 순서 판정

외부 API 응답은 현재 호환성을 위해 딕셔너리로 직렬화하되, 내부 모듈
사이에서는 가능한 한 이 모델을 사용한다.

### `image_processing.py`

이미지 디코딩, RGB 변환, 최대 크기 조정, crop, JPEG·PNG 변환 및 data URL
인코딩을 담당한다. 이미지 경계를 벗어나지 않도록 좌표를 제한하는 기본
연산도 이곳에 둔다.

### `yolo/yolo_inference.py`

기존 `yolo_client.py`의 역할을 옮긴다. 모델 경로와 환경설정, 모델 로딩과
캐싱, 실제 Ultralytics 추론을 담당한다. 애플리케이션 정책에 따른 영역
필터링은 수행하지 않는다.

### `yolo/yolo_processing.py`

YOLO 원시 prediction을 `DetectedRegion`으로 변환한다. polygon 및 bounding
box 계산, 이미지 경계 제한, 좌우 정렬, 너비·면적 조건 필터링을 담당한다.
필터 옵션은 이 모듈에서 정규화한다.

### `ocr/ocr_inference.py`

기존 `ocr_client.py`의 역할을 옮긴다. Google Vision 인증과 요청, 응답을
기본 OCR 토큰으로 변환하는 작업을 담당한다.

### `ocr/ocr_processing.py`

검출 영역의 정방향·90도 회전 crop을 포함하는 contact sheet를 생성한다.
OCR 토큰을 행과 방향에 매핑하고, 방향별 결과를 비교하여 최종 텍스트를
선택한다.

### `call_number_processing.py`

청구기호 정규화, 정렬 키 생성, 기대 순위 계산 및 현재 위치 상태 판정을
담당한다. 이미지나 외부 OCR 서비스에 의존하지 않는다.

### `result_rendering.py`

최종 책등 annotation, YOLO 영역 오버레이, OCR 토큰 오버레이 등 모든
이미지 렌더링을 담당한다. 파일 저장 위치나 API 응답 형식은 알지 못한다.

### `debug_artifacts.py`

파이프라인에서 전달받은 중간 이미지와 메타데이터를 수집한다. 디버그
옵션이 활성화되면 동일한 자료를 다음 두 대상으로 내보낸다.

- 기존 API 응답의 `debug` payload
- 실행별 로컬 디렉터리의 이미지 및 JSON 파일

분석 결과를 계산하거나 변경하지 않는다.

## 데이터 흐름

```text
업로드 이미지
  → 이미지 준비
  → YOLO 추론
  → YOLO prediction 변환·필터링
  → OCR contact sheet 생성
  → OCR 추론
  → OCR 행·방향 매핑 및 텍스트 선택
  → 청구기호 순서 판정
  → 결과 이미지 및 API 응답 생성
```

파이프라인은 각 주요 처리 뒤의 결과를 디버그 수집기에 전달한다. 처리
모듈은 디버그 저장 여부나 저장 경로를 알지 못한다.

## 디버그 산출물

디버그가 비활성화되면 로컬 파일을 생성하지 않는다. 활성화되면 실행마다
충돌하지 않는 ID를 만들고 다음 형태로 저장한다.

```text
debug_runs/
  <run-id>/
    manifest.json
    01-original.jpg
    02-yolo-predictions.jpg
    03-yolo-filtered.jpg
    04-ocr-contact-sheet.png
    05-ocr-bounding-boxes.png
    06-final-result.jpg
```

`manifest.json`에는 실행 ID, 생성 시각, 필터 옵션, 모델 식별자, 원시 검출
수, 필터 통과·탈락 수, OCR 행 정보, 오류가 발생한 단계가 포함된다.

로컬 디버그 루트는 기본적으로 프로젝트의 `debug_runs/`를 사용하되 환경
변수로 변경할 수 있게 한다. `debug_runs/`는 Git 추적 대상에서 제외한다.

## 오류 처리

- 업로드 형식과 크기 검증은 계속 `server.py`가 담당한다.
- 이미지 디코딩 실패는 입력 오류로 처리할 수 있도록 `ValueError` 계열로
  전달한다.
- 모델 파일, 인증 정보, 외부 OCR 오류는 원인 메시지를 보존한다.
- 파이프라인 오류에는 가능하면 실패한 책임 영역을 함께 표시한다.
- 디버그 파일 저장 실패가 정상 분석 결과까지 무효화하지 않게 한다.
  저장 실패 정보는 debug metadata에 기록한다.
- 배치 분석에서는 한 이미지의 실패가 다른 이미지 분석을 중단하지 않는다.

## 테스트 전략

리팩터링 전 현재 API 결과 구조를 특성 테스트로 고정한다. 이후 다음
순서로 책임별 테스트를 구성한다.

- `image_processing`: 크기 조정, 인코딩, 경계 crop
- `yolo_processing`: box·polygon 변환, 정렬, 필터 조건
- `ocr_processing`: contact sheet 배치, 토큰 매핑, 방향 선택
- `call_number_processing`: 정렬 키와 위치 판정
- `result_rendering`: 출력 크기와 기본 이미지 생성
- `debug_artifacts`: 파일명, manifest, 디버그 비활성 시 무출력
- `analysis_pipeline`: YOLO와 OCR 추론을 대역으로 교체한 전체 흐름
- API 회귀 테스트: 단일·배치·테스트 이미지 endpoint의 응답 호환성

실제 YOLO 모델과 Google Vision을 사용하는 테스트는 단위 테스트와
분리한다.

## 마이그레이션 원칙

1. 현재 동작을 나타내는 테스트를 먼저 추가한다.
2. 데이터 모델과 순수 처리 함수를 이동한다.
3. YOLO와 OCR 추론 코드를 각 패키지로 이동한다.
4. 파이프라인을 얇게 구성하고 공개 import 경로를 유지한다.
5. 렌더링과 디버그 저장을 분리한다.
6. 기존 `analysis.py`, `yolo_client.py`, `ocr_client.py`는 모든 참조가
   이동한 뒤 제거한다.
7. 구조 리팩터링이 완료된 후에만 별도 작업으로 알고리즘을 변경한다.

## 완료 기준

- `server.py`가 `from app.analysis import analyze_shelf_photo`로 분석을 호출한다.
- 기존 API의 정상 응답과 디버그 payload가 호환된다.
- YOLO·OCR 후처리를 외부 모델이나 API 없이 독립 테스트할 수 있다.
- 디버그 활성화 시 API payload와 실행별 로컬 산출물이 함께 생성된다.
- 디버그 비활성화 시 로컬 산출물이 생성되지 않는다.
- 기존 단일, 배치, 테스트 이미지 분석 경로가 모두 동작한다.
