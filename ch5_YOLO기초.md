# 5장 YOLO 기초 예제

> Python 3.12 환경에서 가장 권장되는 최신 패키지인 `ultralytics`를 활용한 실시간 객체 탐지 예제다.  
> 실행 시 PyTorch, OpenCV 등 의존성 라이브러리가 함께 설치되며, 공식 경량 모델인 `yolov8n.pt` 파일이 프로젝트 폴더에 자동으로 다운로드되어 실행된다. <br>3~4장에서 다룬 OpenCV 카메라 처리 흐름 위에 YOLO 모델 추론 한 단계만 추가되는 구조라는 점이 이 장의 핵심이다.

```
pip install ultralytics
```

###  `yolo.py`

```python
import cv2
from ultralytics import YOLO

# 1. 미리 훈련된(Pre-trained) YOLOv8 Nano 모델 로드 (가장 가볍고 빠름)
# 실행 시 로컬에 파일이 없으면 인터넷에서 자동으로 다운로드합니다.
model = YOLO("yolov8n.pt")

# 2. 카메라 열기 (실시간 웹캠 기반 객체 탐지 적용)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("웹캠을 실행할 수 없습니다.")
    exit()

print("YOLOv8 실시간 객체 탐지를 시작합니다. 종료하려면 'q'를 누르세요.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 3. YOLO 모델에 현재 프레임 입력하여 추론 (Inference)
    # stream=True를 사용하면 엣지 기기에서 메모리를 아끼며 연속 프레임을 처리합니다.
    results = model(frame, stream=True)

    # 4. 결과 도출 및 시각화
    for r in results:
        # r.plot()은 감지된 물체에 바운딩 박스와 클래스 이름 표기를 자동으로 얹은 이미지를 반환합니다.
        annotated_frame = r.plot()

    # 5. 화면 출력
    cv2.imshow("YOLOv8 Real-Time Detection", annotated_frame)

    # 'q' 키를 누르면 루프 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("YOLO 탐지 시스템이 정상 종료되었습니다.")

# 자원 해제 및 종료
cap.release()
cv2.destroyAllWindows()
print("카메라 스트림이 안전하게 정지되었습니다.")
```

### 해석

`model(frame, stream=True)`에서 `stream=True`는 결과를 리스트가 아니라 제너레이터로 반환해, 젯슨 나노 같은 엣지 장비에서도 메모리를 아끼며 연속 프레임을 처리할 수 있게 한다.  
`r.plot()`은 탐지된 객체마다 바운딩 박스와 클래스 이름을 자동으로 그려 넣은 이미지를 반환하므로, 별도의 시각화 코드 없이 바로 화면에 출력할 수 있다.

이 예제는 4장과 달리 `cap.release()`와 `cv2.destroyAllWindows()`를 루프 종료 후 호출한다(코드에 두 번 반복되어 있는 점은 원본 실습 자료 그대로다). 4장에서 짚었던 자원 해제 누락 문제를 5장에서 실제로 반영한 형태로 볼 수 있다.

### 코드분석

`yolo.py`는 ultralytics 패키지의 YOLOv8 Nano 모델로 웹캠 프레임에서 실시간 객체 탐지를 수행하는 코드다.  
`opencv_camera.py`의 카메라 루프 골격 위에 YOLO 추론과 결과 시각화 두 단계만 추가된 구조다.

<details> <summary>핵심 요약 (블록별 상세 분석)</summary>

**1. 모델 로드**

```python
model = YOLO("yolov8n.pt")  # 로컬에 없으면 실행 시 자동 다운로드
```

미리 학습된 YOLOv8 Nano(가장 가볍고 빠른 버전) 모델을 불러온다.

**2. 카메라 열기**

```python
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("웹캠을 실행할 수 없습니다.")
    exit()
```

`opencv_camera.py`와 동일한 패턴으로 카메라를 열고, 실패 시 즉시 종료한다.

**3. 추론과 결과 시각화**

```python
results = model(frame, stream=True)  # stream=True: 제너레이터로 반환해 메모리 절약

for r in results:
    annotated_frame = r.plot()  # 바운딩 박스·클래스 이름이 그려진 이미지 반환
```

`stream=True`는 결과를 리스트가 아니라 제너레이터로 반환해 메모리가 적은 엣지 장비에서도 연속 프레임을 처리할 수 있게 한다.  
`r.plot()` 덕분에 별도의 그리기 코드 없이 바로 `cv2.imshow`로 출력할 수 있다.

**4. 종료와 자원 해제**

```python
cap.release()
cv2.destroyAllWindows()
```

루프를 벗어난 뒤 카메라 장치를 반환하고 열린 창을 닫는다.  
원본 코드에는 이 두 호출이 연속으로 두 번 반복되어 있는데(중복이지만 오류는 아님), `opencv_camera.py`(4장)에서 짚었던 자원 해제 누락 문제가 이 코드에서는 실제로 반영되어 있다.

</details>

## 전체 흐름 요약

3~4장의 OpenCV 기초(이미지 읽기·처리 → 카메라 실시간 프레임 처리)가 5장 YOLO 실습에서 카메라 프레임을 다루는 방식과 그대로 이어진다.  
두 장을 연속해서 진행하면 학습 효과가 크다.
