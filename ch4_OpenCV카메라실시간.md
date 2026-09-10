# 4장 OpenCV 카메라 실시간 실습

- 출처: 비전 시각화 기초 실습 원본 자료(4장)
- 관련 소스코드: `opencv_camera.py` (03_소스코드)

> 노트북 웹캠이나 USB 카메라에서 영상을 실시간으로 프레임 단위로 읽어와 화면에 출력하고, 텍스트나 사각형을 오버레이하는 기본 파이프라인 구조를 다룬다. 3장에서 익힌 이미지 처리 흐름이 여기서는 정지 이미지 한 장이 아니라 매 프레임마다 반복 적용된다.

### 코드 전문 — `03_소스코드/opencv_camera.py`

```python
import cv2

# 0번 내장 웹캠 오픈 (외부 카메라는 1, 2 등으로 변경 가능)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 웹캠 연결을 확인하세요.")
    exit()

print("카메라 영상 출력을 시작합니다. 'q' 키를 누르면 종료됩니다.")

while True:
    # 실시간 프레임 읽기 (ret: 성공 여부, frame: 이미지 데이터)
    ret, frame = cap.read()
    if not ret:
        print("프레임을 읽어오지 못했습니다.")
        break

    # 좌우 반전 (거울 모드, 원치 않으면 주석 처리)
    frame = cv2.flip(frame, 1)

    # 단순한 그래픽 처리 추가 (사각 박스 및 실시간 안내 텍스트)
    # cv2.rectangle(이미지, 시작좌표, 끝좌표, 색상(BGR), 선두께)
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)
    # cv2.putText(이미지, 텍스트, 시작좌표, 폰트, 크기, 색상, 두께)
    cv2.putText(frame, "Live Camera - Press 'q' to exit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # 처리된 프레임을 화면에 표시
    cv2.imshow("Edge Camera Stream", frame)

    # 1밀리초 동안 키 입력 대기, 'q' 키가 입력되면 루프 탈출
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 해석

`cap.read()`는 매 프레임마다 호출되는 무한 루프 안에서 실행되며, `cv2.waitKey(1)`의 인자 1은 프레임마다 1밀리초만 키 입력을 대기한다는 뜻이다.  
이 값이 실시간 스트리밍의 체감 프레임 속도에 영향을 준다.

### 주의 — 자원 해제 누락

이 예제는 `cap.release()`와 `cv2.destroyAllWindows()` 호출이 빠져 있다.  
카메라 장치를 계속 점유한 상태로 프로그램이 끝날 수 있으므로, 실습 중에는 5장 YOLO 예제처럼 루프 종료 후 반드시 자원을 해제하는 코드를 추가하도록 짚어준다.

### 코드분석

`opencv_camera.py`는 웹캠에서 프레임을 실시간으로 읽어와 화면에 표시하고, 사각형과 텍스트를 오버레이하는 코드다.  
`opencv.py`의 "읽기 → 처리 → 표시" 흐름을 무한 루프 안에서 매 프레임마다 반복한다는 점이 핵심이다.

<details> <summary>핵심 요약 (블록별 상세 분석)</summary>

**1. 카메라 열기와 예외 처리**

```python
cap = cv2.VideoCapture(0)  # 0번 내장 웹캠, 외부 카메라는 1, 2 등으로 변경

if not cap.isOpened():
    print("카메라를 열 수 없습니다. 웹캠 연결을 확인하세요.")
    exit()
```

`cap.isOpened()`로 카메라가 실제로 열렸는지 확인하고, 실패하면 즉시 종료한다.  
이 확인 없이 `cap.read()`를 호출하면 원인을 알기 어려운 오류가 날 수 있다.

**2. 실시간 프레임 읽기 루프**

```python
while True:
    ret, frame = cap.read()  # ret: 성공 여부, frame: 이미지 데이터
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # 좌우 반전 (거울 모드)
```

`cap.read()`는 `(ret, frame)`을 반환한다.  
`ret`이 `False`면(카메라 연결 끊김 등) 루프를 빠져나간다.

**3. 오버레이 그리기**

```python
# cv2.rectangle(이미지, 시작좌표, 끝좌표, 색상(BGR), 선두께)
cv2.rectangle(frame, (100, 100), (300, 300), (0, 255, 0), 3)
# cv2.putText(이미지, 텍스트, 시작좌표, 폰트, 크기, 색상, 두께)
cv2.putText(frame, "Live Camera - Press 'q' to exit", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
```

두 함수 모두 `frame`을 직접 수정(in-place)한다.  
`rectangle`은 초록색 사각형을, `putText`는 파란색 안내 문구를 그린다.

**4. 화면 표시와 종료 조건**

```python
cv2.imshow("Edge Camera Stream", frame)
if cv2.waitKey(1) & 0xFF == ord('q'):  # 1밀리초만 대기 (실시간 체감 속도에 영향)
    break
```

`waitKey(1)`은 `opencv.py`의 `waitKey(0)`(무한 대기)과 달리 짧게 대기하고 바로 다음 프레임으로 넘어가야 실시간 스트리밍처럼 보인다.

주의: 이 코드는 `cap.release()`와 `cv2.destroyAllWindows()` 호출이 없다(원본 그대로). `yolo.py`는 루프 종료 후 이 두 호출을 명시적으로 넣고 있으므로, 실습 중에는 학습자가 직접 추가해보게 하는 것이 좋다.

</details>

## 다음 장과의 연결

실시간 카메라 프레임을 다루는 방식을 익혔다면, 5장에서는 이 프레임에 YOLO 객체 탐지 모델을 적용해 실시간으로 물체를 인식한다. 3~4장의 흐름이 그대로 5장의 카메라 입력 처리로 이어진다.
