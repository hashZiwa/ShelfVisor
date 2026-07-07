# ShelfVisor

ShelfVisor는 도서관 서가 사진을 업로드하면 이미지 안의 책 관련 영역을 탐지하고, 이후 OCR과 청구기호 정렬 검증으로 이어가기 위한 웹 서비스 데모입니다.

## 현재 구성

현재 브랜치는 FastAPI 서버와 로컬 YOLO 모델 기반 prediction 로직으로 동작합니다.

- **FastAPI**: 이미지 업로드 API와 웹 화면 제공
- **로컬 YOLO 모델**: `models/yolo/yolo-model-v1.pt` 파일을 사용해 라벨 후보 영역 탐지
- **Pillow**: 이미지 변환과 결과 이미지 생성
- **Mock OCR**: 실제 OCR 연동 전까지 임시 청구기호 생성
- **브라우저 UI**: 사진 업로드, 테스트 이미지 실행, 결과 이미지와 디버그 단계 표시

## 분석 흐름

1. 서가 사진을 업로드합니다.
2. 서버가 이미지를 적당한 크기로 축소하고 JPEG로 변환합니다.
3. `models/yolo/yolo-model-v1.pt` 로컬 YOLO 모델로 prediction을 수행합니다.
4. prediction의 `x`, `y`, `width`, `height`, `confidence`, `class` 값을 ShelfVisor 내부 영역 형식으로 변환합니다.
5. YOLO prediction을 이미지 위에 표시한 디버그 이미지를 생성합니다.
6. Mock OCR 청구기호와 결과 표시 이미지를 반환합니다.

## 로컬 YOLO 설정

기본 모델 경로는 아래와 같습니다.

```text
models/yolo/yolo-model-v1.pt
```

필요하면 `.env` 파일에서 경로와 추론 설정을 바꿀 수 있습니다.

```powershell
LOCAL_YOLO_MODEL_PATH=models/yolo/yolo-model-v1.pt
LOCAL_YOLO_CONFIDENCE=0.15
LOCAL_YOLO_IMAGE_SIZE=1024
```

## 설치

가상환경 사용을 권장합니다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

로컬 YOLO 추론에는 `ultralytics`와 그 의존성인 `torch`가 필요합니다. 설치에 시간이 걸릴 수 있습니다.

## 실행

```powershell
python app/server.py
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:8000
```

FastAPI 문서는 아래에서 확인할 수 있습니다.

```text
http://127.0.0.1:8000/docs
```

## API

### 상태 확인

```text
GET /api/health
```

### 이미지 분석

```text
POST /api/analyze
```

요청 형식은 `multipart/form-data`이며, 이미지 파일 필드 이름은 `image`입니다.

### 복수 이미지 분석

```text
POST /api/analyze-batch
```

이미지 파일 필드 이름은 `images`입니다. 현재 최대 5장까지 처리합니다.

## 개발 메모

- 현재 추가된 로컬 YOLO 모델은 책등의 청구기호 라벨 검출 모델입니다.
- 현재 파이프라인은 YOLO prediction을 바로 결과 영역으로 사용합니다.
- 다음 단계에서는 라벨 YOLO 결과를 OCR crop 후보로 직접 넘기는 흐름과, 책등 검출 흐름을 분리하는 것이 좋습니다.
