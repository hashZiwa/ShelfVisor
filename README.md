# ShelfVisor

ShelfVisor는 도서관 서가 사진을 업로드하면 사진 속 책들이 올바른 순서로 배가되어 있는지 확인하는 웹 서비스 데모입니다.

## 현재 구성

현재 `YOLO-api-v1` 브랜치는 FastAPI 기반 서버와 Roboflow YOLO API 기반 책등 탐지 로직으로 동작합니다.

- **FastAPI**: 이미지 업로드 API와 웹 화면 제공
- **Roboflow Serverless API**: `book-spine-detection-2cci9/2` 모델로 책등 탐지
- **Pillow**: 이미지 변환과 결과 이미지 생성
- **Mock OCR**: 실제 OCR 연동 전 단계의 임시 청구 기호 생성
- **정렬 규칙 검사**: 왼쪽에서 오른쪽으로 청구 기호가 정렬되어 있는지 확인
- **브라우저 UI**: 서가 사진 업로드, 분석 결과 이미지 표시, 청구 기호 목록 표시

## 서버와 분석 방식

이 서비스는 이미지 분석과 OCR 연동이 핵심이므로 Python API 서버 방식이 잘 맞습니다. FastAPI는 업로드 API, 분석 결과 JSON 응답, 자동 API 문서, 향후 비동기 작업 전환에 적합합니다.

책등 구분은 현재 Roboflow에서 제공하는 YOLO 모델 API를 호출해 테스트합니다. API 키는 코드에 저장하지 않고 환경변수 또는 `.env`로 주입합니다.

1. 업로드 이미지를 적당한 크기로 축소합니다.
2. 이미지를 JPEG로 변환합니다.
3. Roboflow Serverless API에 base64 이미지 데이터를 전송합니다.
4. YOLO prediction의 `x`, `y`, `width`, `height`, `confidence`, `class` 값을 읽습니다.
5. prediction을 기존 ShelfVisor 응답 형식의 책등 영역으로 변환합니다.
6. 감지된 영역을 이미지 위에 표시합니다.

## Roboflow 설정

`.env.example`을 참고해 `.env` 파일을 만들고 API 키를 설정합니다.

```powershell
ROBOFLOW_API_KEY=your_roboflow_api_key
ROBOFLOW_API_URL=https://serverless.roboflow.com
ROBOFLOW_MODEL_ID=book-spine-detection-2cci9/2
```

현재 Python 3.13 환경에서는 `inference-sdk`가 설치되지 않으므로, 이 브랜치는 SDK 대신 동일한 Serverless endpoint를 직접 HTTP로 호출합니다.

## 설치

가상환경 사용을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## 실행

```powershell
python app/server.py
```

실행 후 브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8000
```

FastAPI 문서는 아래 주소에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

## API

### 상태 확인

```text
GET /api/health
```

### 서가 이미지 분석

```text
POST /api/analyze
```

요청 형식은 `multipart/form-data`이며, 이미지 파일 필드 이름은 `image`입니다.

응답에는 감지된 책등 목록, 청구 기호, 정렬 상태, 주석이 표시된 이미지가 포함됩니다.

## 현재 데모 흐름

1. 서가 사진을 업로드합니다.
2. 서버가 Roboflow YOLO API로 책등 후보 영역을 추정합니다.
3. 감지된 책등마다 임시 청구 기호를 생성합니다.
4. 데모에서 순서 오류를 확인할 수 있도록, 책이 충분히 많을 때 일부 청구 기호 순서를 의도적으로 바꿉니다.
5. 감지 영역, OCR 유사 결과, 정렬 상태, 주석이 표시된 이미지를 JSON으로 반환합니다.

## 디버그 모드

화면 상단의 **디버그 보기**를 켜고 사진을 업로드하면 OpenCV 처리 중간 이미지를 함께 확인할 수 있습니다.

디버그 단계는 아래 순서로 표시됩니다.

1. 원본
2. 분석 영역
3. 전처리
4. 세로 에지
5. 경계 후보
6. 최종 책등 박스

API로 직접 호출할 때는 `multipart/form-data`에 `debug=true` 필드를 함께 보내면 됩니다.

## 다음 개발 단계

- 실제 서가 사진 샘플을 모아 Roboflow 모델 탐지 품질 확인
- prediction confidence threshold와 후처리 규칙 추가
- `mock_ocr_call_numbers`를 Google Cloud Vision OCR 연동으로 교체
- 청구 기호 체계별 정규화 및 정렬 규칙 추가
- OCR 신뢰도가 낮은 항목을 사람이 검토할 수 있는 화면 추가
- 분석 이력 저장과 결과 리포트 내보내기 기능 추가
