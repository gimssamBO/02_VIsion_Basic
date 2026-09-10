# 2장 CNN 모델 전 과정 기초 예제

- 저장된 모델 파일: `my_cnn_model.keras`, `my_cnn_model.pth`

> 합성곱 신경망(CNN)은 이미지에서 특징(edge, texture 등)을 추출하는 합성곱(Conv) 층과, <br>특징 맵의 크기를 줄이는 풀링(Pooling) 층을 반복해 쌓은 뒤, <br>마지막에 완전연결층(Dense)으로 분류를 수행하는 구조다. <br>손글씨 숫자 데이터셋 MNIST를 이용해 TensorFlow와 PyTorch 두 프레임워크로 <br>각각 데이터 로드부터 모델 저장·재적용까지 전 과정을 구현한다.

### CNN 구조 개념

- 합성곱(Conv) 층: 이미지 위를 필터가 훑으며 edge, texture 같은 특징을 추출한다.
- 풀링(Pooling) 층: 특징 맵의 크기를 줄여 연산량을 줄이고 핵심 특징만 남긴다.
- 완전연결(Dense) 층: 추출된 특징을 바탕으로 최종 클래스를 분류한다.

## TensorFlow vs PyTorch 구조 차이

같은 CNN을 구현해도 두 프레임워크의 작성 방식과 저장 방식은 다르다. <br>실습 전에 이 차이를 먼저 이해해야 두 코드를 비교하며 볼 수 있다.

| 구분 | TensorFlow (Keras) | PyTorch |
| --- | --- | --- |
| 학습 루프 | `model.fit()` 한 줄로 자동 처리 | `forward → loss → backward → step`을 직접 작성 |
| 모델 저장 형식 | `.keras` 파일(모델 구조 + 가중치) | `.pth` 파일(가중치 `state_dict`만) |
| 모델 로드 | `load_model()`로 구조까지 복원 | 동일한 모델 클래스를 먼저 생성한 뒤 `load_state_dict()`로 가중치만 덮어씀 |
| GPU 사용 | 자동 우선 사용(1장 참고) | `.to(device)`로 명시적 이동 필요(1장 참고) |

## TensorFlow 구현

### 코드 전문 — `03_소스코드/cnn_tensorflow.py`

```python
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. 데이터 셋 로드 (Kaggle 대신 가볍게 사용 가능한 MNIST 내장 데이터셋 활용)
print("--- 1. 데이터 로드 ---")
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 데이터 전처리: 픽셀 값 정규화 (0~255 -> 0~1) 및 채널 차원 추가 (CNN 입력 형태)
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255

# 2. 모델 생성
print("\n--- 2. 모델 생성 ---")
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 0~9까지 10개 클래스 분류
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# 3. 모델 훈련
print("\n--- 3. 모델 훈련 ---")
model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_split=0.1)

# 4. 모델 평가
print("\n--- 4. 모델 평가 ---")
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")

# 5. 모델 저장 (Keras 표준 양식인 .keras 파일로 저장)
print("\n--- 5. 모델 저장 ---")
model.save("my_cnn_model.keras")
print("모델이 'my_cnn_model.keras'로 저장되었습니다.")

# 6. 모델 로드
print("\n--- 6. 모델 로드 ---")
loaded_model = tf.keras.models.load_model("my_cnn_model.keras")
print("저장된 모델을 성공적으로 불러왔습니다.")

# 7. 모델 적용 및 결과 도출 (테스트 데이터 중 1개 샘플로 예측)
print("\n--- 7. 모델 적용 및 결과 도출 ---")
sample_image = test_images[0]        # (28, 28, 1) 형태
sample_label = test_labels[0]        # 실제 정답

# 예측을 위해 배치를 위한 차원 추가: (28, 28, 1) -> (1, 28, 28, 1)
input_data = np.expand_dims(sample_image, axis=0)

predictions = loaded_model.predict(input_data)
predicted_class = np.argmax(predictions[0])

print(f"실제 정답 (Label): {sample_label}")
print(f"모델 예측 결과 (Prediction): {predicted_class}")
```

**실행결과**
```
--- 1. 데이터 로드 ---
Downloading data from [https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
11490434/11490434 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

--- 2. 모델 생성 ---

/usr/local/lib/python3.13/dist-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)

Model: "sequential"

┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv2d (Conv2D)                 │ (None, 26, 26, 32)     │           320 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 13, 13, 32)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 11, 11, 64)     │        18,496 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 5, 5, 64)       │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ flatten (Flatten)               │ (None, 1600)           │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 64)             │       102,464 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 10)             │           650 │
└─────────────────────────────────┴────────────────────────┴───────────────┘

 Total params: 121,930 (476.29 KB)

 Trainable params: 121,930 (476.29 KB)

 Non-trainable params: 0 (0.00 B)

--- 3. 모델 훈련 ---
Epoch 1/3
844/844 ━━━━━━━━━━━━━━━━━━━━ 10s 7ms/step - accuracy: 0.9449 - loss: 0.1896 - val_accuracy: 0.9847 - val_loss: 0.0537
Epoch 2/3
844/844 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9827 - loss: 0.0552 - val_accuracy: 0.9882 - val_loss: 0.0444
Epoch 3/3
844/844 ━━━━━━━━━━━━━━━━━━━━ 3s 4ms/step - accuracy: 0.9873 - loss: 0.0384 - val_accuracy: 0.9885 - val_loss: 0.0453

--- 4. 모델 평가 ---
313/313 - 2s - 6ms/step - accuracy: 0.9880 - loss: 0.0343
테스트 정확도: 0.9880

--- 5. 모델 저장 ---
모델이 'my_cnn_model.keras'로 저장되었습니다.

--- 6. 모델 로드 ---
저장된 모델을 성공적으로 불러왔습니다.

--- 7. 모델 적용 및 결과 도출 ---
1/1 ━━━━━━━━━━━━━━━━━━━━ 0s 413ms/step
실제 정답 (Label): 7
모델 예측 결과 (Prediction): 7
```
### 해석

`model.compile`에서 손실 함수로 `sparse_categorical_crossentropy`를 쓰는 이유는 레이블이 원-핫 인코딩이 아니라 정수(0~9) 형태이기 때문이다. 마지막 층의 `softmax` 활성화 함수는 10개 클래스 각각에 대한 확률을 출력하며, 예측 시에는 `np.argmax`로 가장 확률이 높은 클래스를 선택한다.

`model.save("my_cnn_model.keras")`를 실행하면 프로젝트 폴더에 `my_cnn_model.keras` 파일이 생성된다. 이 파일에는 모델 구조와 학습된 가중치가 함께 저장되어 있어, 이후 `tf.keras.models.load_model()`로 그대로 불러와 6~7단계의 로드·예측 과정에 사용된다.

### 코드분석

`cnn_tensorflow.py`는 MNIST 데이터로 CNN을 만들어 학습·평가·저장·재적용까지 전 과정을 처리하는 코드다. 데이터 로드 → 모델 생성 → 컴파일 → 훈련 → 평가 → 저장 → 로드 → 예측, 7단계로 구성되며, `model.fit()` 한 줄로 학습 루프 전체를 처리하는 Keras의 고수준 API 방식이 핵심이다.

<details>
<summary>핵심 요약 (블록별 상세 분석)</summary>

**1. 데이터 로드 및 전처리**

```python
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# 픽셀 값 정규화(0~255 -> 0~1) 및 채널 차원 추가(CNN 입력 형태)
train_images = train_images.reshape((60000, 28, 28, 1)).astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype("float32") / 255
```

6만 장(학습)·1만 장(테스트)의 28x28 흑백 손글씨 숫자 이미지를 내려받는다. `reshape`으로 채널 차원(흑백=1)을 추가하는 이유는 `Conv2D` 층이 채널 차원을 입력으로 기대하기 때문이다.

**2. 모델 생성**

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 0~9까지 10개 클래스 분류
])
```

`Conv2D → MaxPooling2D` 패턴이 두 번 반복된 뒤, `Flatten`으로 1차원으로 펼치고 `Dense(64)` → `Dense(10, softmax)`로 최종 10개 클래스 확률을 출력한다.

**3. 컴파일과 요약**

```python
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # 레이블이 정수(0~9)이므로 sparse 버전 사용
              metrics=['accuracy'])
model.summary()
```

레이블이 원-핫 인코딩이 아니라 정수(0~9)이므로 `sparse_categorical_crossentropy`를 쓴다. `model.summary()`는 층별 출력 shape와 파라미터 개수를 표로 보여준다.

**4. 훈련과 평가**

```python
model.fit(train_images, train_labels, epochs=3, batch_size=64, validation_split=0.1)

test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"테스트 정확도: {test_acc:.4f}")
```

`model.fit()` 한 줄이 순전파·손실 계산·역전파·가중치 갱신을 3 에폭 동안 자동으로 반복한다. `validation_split=0.1`로 학습 데이터의 10%를 검증용으로 떼어낸다.

**5. 저장, 로드, 예측**

```python
model.save("my_cnn_model.keras")  # 모델 구조 + 가중치를 함께 저장

loaded_model = tf.keras.models.load_model("my_cnn_model.keras")

# 배치 차원 추가: (28,28,1) -> (1,28,28,1)
input_data = np.expand_dims(sample_image, axis=0)
predictions = loaded_model.predict(input_data)
predicted_class = np.argmax(predictions[0])
```

`model.save()`는 모델 구조와 가중치를 `.keras` 파일 하나에 함께 저장한다. 예측 시 `np.expand_dims`로 배치 차원을 추가하는 이유는 모델이 항상 배치 차원을 첫 번째 축으로 기대하기 때문이다.

</details>

## PyTorch 구현

### 코드 전문 — `03_소스코드/cnn_pytorch.py`

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np

# 컴퓨터의 GPU 사용 가능 여부 확인 (젯슨 나노 등 엣지 장비나 외장 그래픽이 있다면 cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"현재 연산 장치: {device}\n")

# 1. 데이터 셋 로드 (Kaggle 대신 가볍게 사용 가능한 MNIST 내장 데이터셋 활용)
print("--- 1. 데이터 로드 및 전처리 ---")
# 전처리 정의: 이미지를 PyTorch 텐서로 변환하고 0~1 사이로 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # -1 ~ 1 사이로 정규화
])

# 데이터셋 다운로드 및 로드
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 배치를 만들기 위한 데이터 로더 (Data Loader) 생성
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 2. 모델 생성 (CNN 구조 정의)
print("\n--- 2. 모델 생성 ---")
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 입력 채널: 1 (흑백), 출력 채널: 32, 커널 크기: 3x3
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2 맥스풀링
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 이미지 크기 변화: 28x28 -> pool -> 14x14 -> pool -> 7x7
        # 64개 채널 x 7x7 크기 = 3136 차원을 1차원으로 펼침
        self.fc1 = nn.Linear(64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 10)  # 최종 10개 클래스 분별
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten (1차원 배열로 펼치기)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # PyTorch CrossEntropyLoss를 쓸 때는 마지막에 Softmax를 생략합니다.
        return x

model = SimpleCNN().to(device)
print(model)

# 손실 함수와 최적화 도구(Optimizer) 정의
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 3. 모델 훈련
print("\n--- 3. 모델 훈련 ---")
epochs = 3
for epoch in range(epochs):
    model.train()  # 모델을 훈련 모드로 설정
    running_loss = 0.0
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)

        # 역전파 전 그라디언트 초기화
        optimizer.zero_grad()

        # 순전파 (Forward) 및 손실 계산
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 역전파 (Backward) 및 가중치 업데이트
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# 4. 모델 평가
print("\n--- 4. 모델 평가 ---")
model.eval()  # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 비활성화)
correct = 0
total = 0

with torch.no_grad():  # 평가 시에는 기울기(Gradient) 계산을 안 함 (메모리 절약)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 가장 높은 확률을 가진 인덱스 추출
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"테스트 정확도: {(correct / total) * 100:.2f}%")

# 5. 모델 저장
print("\n--- 5. 모델 저장 ---")
# PyTorch는 모델의 가중치 상태(state_dict)만 저장하는 것이 표준 관례입니다.
torch.save(model.state_dict(), "my_cnn_model.pth")
print("모델 가중치가 'my_cnn_model.pth'로 저장되었습니다.")

# 6. 모델 로드
print("\n--- 6. 모델 로드 ---")
# 먼저 모델 객체를 똑같이 생성한 뒤 가중치를 덮어씌웁니다.
loaded_model = SimpleCNN().to(device)
loaded_model.load_state_dict(torch.load("my_cnn_model.pth", map_location=device))
print("저장된 모델을 성공적으로 불러왔습니다.")

# 7. 모델 적용 및 결과 도출 (테스트 데이터 중 1개 샘플로 예측)
print("\n--- 7. 모델 적용 및 결과 도출 ---")
loaded_model.eval()

# 테스트 데이터셋에서 무작위로 1개 가져오기
sample_image, sample_label = test_dataset[0]  # sample_image 크기: (1, 28, 28)

# 예측을 위해 배치를 위한 차원 추가 (배치 크기 1): (1, 28, 28) -> (1, 1, 28, 28)
input_data = sample_image.unsqueeze(0).to(device)

with torch.no_grad():
    prediction = loaded_model(input_data)
    # 가장 큰 연산 결과 값을 가진 인덱스가 예측 클래스
    predicted_class = torch.argmax(prediction, dim=1).item()

print(f"실제 정답 (Label): {sample_label}")
print(f"모델 예측 결과 (Prediction): {predicted_class}")
```

**결과 출력**
```
현재 연산 장치: cuda

--- 1. 데이터 로드 및 전처리 ---

100%|██████████| 9.91M/9.91M [00:00<00:00, 18.5MB/s]
100%|██████████| 28.9k/28.9k [00:00<00:00, 494kB/s]
100%|██████████| 1.65M/1.65M [00:00<00:00, 4.58MB/s]
100%|██████████| 4.54k/4.54k [00:00<00:00, 4.30MB/s]

--- 2. 모델 생성 ---
SimpleCNN(
  (conv1): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
  (fc1): Linear(in_features=3136, out_features=64, bias=True)
  (fc2): Linear(in_features=64, out_features=10, bias=True)
  (relu): ReLU()
)

--- 3. 모델 훈련 ---
Epoch [1/3], Loss: 0.1720
Epoch [2/3], Loss: 0.0488
Epoch [3/3], Loss: 0.0351

--- 4. 모델 평가 ---
테스트 정확도: 98.58%

--- 5. 모델 저장 ---
모델 가중치가 'my_cnn_model.pth'로 저장되었습니다.

--- 6. 모델 로드 ---
저장된 모델을 성공적으로 불러왔습니다.

--- 7. 모델 적용 및 결과 도출 ---
실제 정답 (Label): 7
모델 예측 결과 (Prediction): 7
```
### 해석

PyTorch는 TensorFlow와 달리 학습 루프(순전파, 손실 계산, 역전파, 가중치 갱신)를 `for epoch`와 `for i, (images, labels)` 이중 반복문으로 직접 작성한다. `optimizer.zero_grad()`로 이전 그라디언트를 초기화하지 않으면 그라디언트가 누적되어 학습이 잘못되므로, 매 배치마다 반드시 호출해야 한다.

`torch.save(model.state_dict(), "my_cnn_model.pth")`를 실행하면 프로젝트 폴더에 `my_cnn_model.pth` 파일이 생성된다. TensorFlow와 달리 모델 구조는 저장되지 않고 학습된 가중치(state_dict)만 저장되므로, 6단계에서 `SimpleCNN()`으로 동일한 구조의 모델을 먼저 만든 뒤 `load_state_dict()`로 이 파일의 가중치를 덮어써서 사용한다.

### 코드분석

`cnn_pytorch.py`는 같은 MNIST 분류 문제를 PyTorch로 학습 루프를 직접 작성해 구현한 코드다. `forward → loss → backward → step`을 이중 반복문으로 직접 쓴다는 점이 Keras의 `model.fit()`과 가장 크게 다르다.

<details>
<summary>핵심 요약 (블록별 상세 분석)</summary>

**1. 장치 설정과 데이터 준비**

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 이미지를 텐서로 변환 + -1~1 범위로 정규화
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# batch_size 단위로 데이터를 공급, 학습용은 shuffle=True로 매 에폭마다 순서를 섞음
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
```

`transforms.Compose`는 이미지를 텐서로 바꾸고 -1~1 범위로 정규화하는 전처리를 하나로 묶는다. `DataLoader`는 데이터셋을 `batch_size`(64) 단위로 잘라 공급한다.

**2. 모델 정의**

```python
class SimpleCNN(nn.Module):
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)  # Flatten (1차원 배열로 펼치기, TensorFlow의 Flatten과 동일 역할)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)  # PyTorch CrossEntropyLoss를 쓸 때는 마지막에 Softmax를 생략합니다.
        return x
```

`nn.Module`을 상속받아 층은 `__init__`에서, 연산 순서는 `forward`에 작성한다. 마지막 `fc2`가 소프트맥스 없이 점수만 출력하는 이유는 `nn.CrossEntropyLoss`가 소프트맥스를 내부적으로 포함하기 때문이다.

**3. 손실함수·옵티마이저와 학습 루프**

```python
for epoch in range(epochs):
    for i, (images, labels) in enumerate(train_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()              # 1. 이전 배치의 그라디언트 초기화
        outputs = model(images)            # 2. 순전파(Forward)
        loss = criterion(outputs, labels)  # 3. 손실 계산
        loss.backward()                    # 4. 역전파(Backward)
        optimizer.step()                   # 5. 가중치 갱신
```

TensorFlow의 `model.fit()` 한 줄이 여기서는 이중 반복문으로 풀어서 작성된다. `optimizer.zero_grad()`를 호출하지 않으면 그라디언트가 누적되어 학습이 잘못되므로 매 배치마다 반드시 호출해야 한다.

**4. 평가**

```python
model.eval()  # 평가 모드로 전환
with torch.no_grad():  # 불필요한 그라디언트 계산을 막아 메모리 절약
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 가장 높은 확률을 가진 인덱스 추출
```

`model.eval()`은 평가 모드로 전환하고, `torch.no_grad()`는 불필요한 그라디언트 계산을 막아 메모리를 아낀다.

**5. 저장, 로드, 예측**

```python
torch.save(model.state_dict(), "my_cnn_model.pth")  # 가중치(state_dict)만 저장

loaded_model = SimpleCNN().to(device)
loaded_model.load_state_dict(torch.load("my_cnn_model.pth", map_location=device))

# 배치 차원 추가: (1,28,28) -> (1,1,28,28)
input_data = sample_image.unsqueeze(0).to(device)
```

`torch.save(model.state_dict(), ...)`는 모델 구조가 아니라 가중치만 저장한다. 로드할 때는 `SimpleCNN()`으로 동일한 구조를 먼저 만든 뒤 가중치를 덮어써야 한다.

비교: 같은 MNIST 분류 문제를 풀지만, `cnn_tensorflow.py`는 `model.fit()` 한 줄로 학습 루프가 끝나고 `.save()`에 구조까지 통째로 저장된다는 점이 가장 큰 차이다.

</details>

## 다음 장과의 연결

CNN을 통한 이미지 분류를 마쳤다면, 다음은 이미지 자체를 다루는 기초로 넘어간다. <br>3장부터는 OpenCV로 이미지를 읽고 처리하는 흐름을 다룬다.
