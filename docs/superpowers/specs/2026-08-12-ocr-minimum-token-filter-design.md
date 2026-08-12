# OCR Minimum Token Filter Design

## Goal

OCR 토큰 박스가 2개 이하인 방향 이미지를 선택 후보에서 제외한다. 원본과
90도 회전 방향이 모두 제외된 이미지 세트는 `OCR bounding boxes` 디버그
단계에서 탈락 상태를 마지막으로 표시한 뒤 최종 분석에서 완전히 제거한다.

## Eligibility Rules

- 원본과 90도 반시계 회전 방향을 각각 독립적으로 검사한다.
- OCR 토큰 박스가 3개 이상인 방향만 선택 후보로 인정한다.
- 후보가 하나이면 해당 방향을 선택한다.
- 후보가 둘이면 기존과 같이 평균 confidence가 더 높은 방향을 선택한다.
- 두 후보의 평균 confidence가 같으면 기존 방향 순서에 따라 원본을 선택한다.
- 두 방향 모두 토큰 박스가 2개 이하이면 이미지 세트 전체를 탈락시킨다.

## OCR Row Data

OCR 행과 방향 결과에 명시적인 적격 상태를 기록한다.

- 각 `variantResults` 항목에 `eligible` boolean을 추가한다.
- 행에는 `eligible` boolean을 추가한다.
- 통과 행에는 기존과 같이 `selectedOrientation`, `text`, `tokens`,
  `selectedAverageConfidence`를 기록한다.
- 탈락 행은 `eligible=False`, `selectedOrientation=None`, 빈 `text`와 빈
  `tokens`, `selectedAverageConfidence=0.0`을 사용한다.

이 상태는 OCR 추론 결과와 원래 contact sheet 행의 대응 관계를 유지하면서
디버그 렌더링과 후속 필터링에서 같은 판정을 공유하게 한다.

## Debug Rendering

`OCR contact sheet` 단계는 OCR 이전 입력을 보여주므로 기존과 같이 모든
이미지 세트를 표시한다.

`OCR bounding boxes` 단계에서는 모든 OCR 행을 표시한다.

- 토큰이 3개 이상인 방향은 기존 방향 색상 또는 선택된 초록색 테두리를
  사용한다.
- 토큰이 2개 이하인 방향은 선택 여부와 관계없이 붉은색 테두리를 사용한다.
- 두 방향이 모두 탈락한 세트는 양쪽 이미지가 붉은색 테두리로 표시된다.
- 토큰 bounding box와 confidence 캡션은 탈락 방향에도 그대로 표시해 탈락
  근거를 확인할 수 있게 한다.
- 디버그 상세 문자열에는 각 방향의 `eligible` 또는 `rejected` 상태와 행
  전체의 탈락 상태를 포함한다.

## Pipeline Filtering

OCR 매핑 직후에는 모든 OCR 행과 모든 YOLO 영역을 유지한다. 먼저 전체 OCR
행으로 `OCR bounding boxes` 디버그 이미지를 생성한다. 이후 OCR 행과 YOLO
영역을 위치별로 함께 묶고, `row["eligible"]`이 참인 쌍만 후속 분석에 넘긴다.

통과 쌍만 사용해 다음을 계산한다.

- 청구기호 목록과 정렬 상태
- 최종 `spines`
- `summary.bookCount`와 `summary.misplacedCount`
- 최종 결과 이미지

최종 spine 번호는 통과 항목만 대상으로 1부터 연속으로 다시 부여한다. 원래
OCR 행의 `index`는 `OCR bounding boxes` 디버그 단계에서 원본 세트 식별을
위해 유지한다.

## Data Flow

1. 모든 YOLO 영역으로 OCR contact sheet를 생성한다.
2. OCR annotation을 각 행과 방향에 매핑하고 토큰을 정렬한다.
3. 방향별 토큰 수로 적격 여부를 계산한다.
4. 적격 방향 중 최종 방향을 선택하거나 행 전체를 탈락시킨다.
5. 전체 행으로 붉은 탈락 표시를 포함한 `OCR bounding boxes`를 생성한다.
6. 통과한 OCR 행과 대응 YOLO 영역만 함께 추출한다.
7. 통과 항목만으로 정렬 검사, spine 생성, 요약, 최종 이미지를 만든다.

## Edge Cases

- 토큰 수가 정확히 2개이면 탈락한다.
- 토큰 수가 정확히 3개이면 통과한다.
- 한 방향만 통과하면 낮은 confidence여도 그 방향을 선택한다.
- 모든 이미지 세트가 탈락하면 `bookCount=0`, 빈 `spines`, 빈 정렬 입력을
  사용하며 최종 이미지는 annotation이 없는 원본 이미지가 된다.
- OCR 행과 YOLO 영역은 contact sheet 생성 시 동일 순서를 사용하므로 위치별
  결합으로 대응시킨다.

## Testing

OCR 후처리 단위 테스트:

- 2개 토큰 방향은 탈락하고 3개 토큰 방향은 선택된다.
- 두 방향이 모두 3개 이상이면 평균 confidence로 선택한다.
- 두 방향이 모두 2개 이하이면 행 전체가 탈락한다.
- 탈락 행은 선택 방향과 최종 토큰·텍스트를 갖지 않는다.

렌더링 단위 테스트:

- 탈락 방향의 이미지 테두리가 붉은색이다.
- 한 방향만 탈락한 세트에서는 탈락 방향만 붉고 통과·선택 방향은 기존
  초록색 강조를 유지한다.

파이프라인 테스트:

- 탈락 세트가 `OCR bounding boxes` 디버그 데이터에는 남는다.
- 탈락 세트가 최종 `spines`와 `bookCount`에서는 제거된다.
- 살아남은 spine 번호가 1부터 연속으로 다시 부여된다.
- 전체 회귀 테스트로 기존 디버그 단계 순서와 응답 구조를 확인한다.

## Scope

이번 변경에는 OCR API 호출 방식, 토큰 정렬 규칙, 평균 confidence 계산식,
YOLO 크기 필터, contact sheet 레이아웃 변경이 포함되지 않는다.
