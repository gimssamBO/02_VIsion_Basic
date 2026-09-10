# 1장 GPU 설정 작업

## 1-1. GPU 있는 컴퓨터에서 설정

> 딥러닝 연산은 CPU보다 GPU에서 훨씬 빠르다. PyTorch는 사용자가 연산 장치를 직접 지정해야 하고, TensorFlow는 GPU가 있으면 자동으로 우선 사용한다는 점이 핵심 차이다. 2장부터 다룰 CNN 학습 속도가 이 장에서 GPU 인식이 제대로 되었는지에 따라 크게 달라지므로, 실습 전에 반드시 짚고 넘어가야 한다.

## PyTorch 환경 — 장치를 직접 지정하는 방식

PyTorch는 사용자가 연산할 장치(device)를 직접 지정하여 모델과 데이터를 그 장치로 보내야 한다.

구글 코랩에서 그래픽 드라이버 설치 상태를 확인한다.
https://colab.research.google.com/drive/1YNz3snUdhmZgT0HU-LgGr5cn5BlJRd9E?hl=ko#scrollTo=venHLMvn8P8M

```
nvidia-smi
```

실행 결과 예시:

```
Sat Aug 15 16:28:16 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 596.21                 Driver Version: 596.21         CUDA Version: 13.2     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA GeForce RTX 3070      WDDM  |   00000000:01:00.0  On |                  N/A |
| 33%   29C    P8             10W /  220W |    1214MiB /   8192MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

그래픽카드 자체는 RTX 3070으로 정상 인식되지만, 드라이버가 보는 CUDA 버전(13.2)과 실제로 설치된 CUDA Toolkit 버전이 다를 수 있다는 점에 주의한다. 

필요하면 아래 명령으로 CUDA를 설치한다.

```
winget install -e --id Nvidia.CUDA
```

PyTorch가 실제로 CUDA를 인식하는지는 아래 한 줄 명령으로 즉시 확인할 수 있다.

```
python -c "import torch; print('CUDA 가용 여부:', torch.cuda.is_available()); print('장치 개수:', torch.cuda.device_count()); print('장치 이름:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '없음')"
```

이번 실습 환경에서의 실행 결과:

```
CUDA 가용 여부: False
장치 개수: 0
장치 이름: 없음
```

`nvidia-smi`로 GPU 하드웨어는 확인되지만 PyTorch는 CUDA를 인식하지 못하는 상태다. 이는 CPU 전용(CPU-only) 빌드의 PyTorch가 설치되었을 때 흔히 나타나는 증상이다. 이 경우 아래처럼 삭제 후 CUDA 지원 버전으로 재설치한다.

```
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio
```


## 1-2. GPU없이 구글 코랩에서 실행

https://colab.research.google.com/drive/1YNz3snUdhmZgT0HU-LgGr5cn5BlJRd9E?hl=ko#scrollTo=venHLMvn8P8M
### PyTorch 환경 

PyTorch는 사용자가 연산할 장치(device)를 직접 지정하여 모델과 데이터를 그 장치로 보내야 함.

### 그래픽 드라이버 설치 상태를 확인.
```
!nvidia-smi
```
실행 결과
```
Wed Sep  9 16:15:44 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 580.82.07              Driver Version: 580.82.07      CUDA Version: 13.0     |
+-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  Tesla T4                       Off |   00000000:00:04.0 Off |                    0 |
| N/A   40C    P8              9W /   70W |       0MiB /  15360MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+

+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
```
### CUDA 설치

```python
# Check CUDA Toolkit version
!nvcc --version

# Check PyTorch CUDA availability
import torch
print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch CUDA version: {torch.version.cuda}")

# Check TensorFlow GPU availability
import tensorflow as tf
print(f"TensorFlow GPU available: {tf.config.list_physical_devices('GPU')}")
```
실행 결과
```
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Fri_Feb_21_20:23:50_PST_2025
Cuda compilation tools, release 12.8, V12.8.93
Build cuda_12.8.r12.8/compiler.35583870_0
PyTorch CUDA available: True
PyTorch CUDA version: 12.8
TensorFlow GPU available: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### `03_소스코드/pytorch_check.py`

```python
import torch

# 1. GPU 장치 존재 확인 및 정의
# GPU(CUDA)가 있으면 'cuda'를 사용하고, 없으면 'cpu'를 자동으로 선택하는 코드입니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"현재 선택된 하드웨어 장치: {device}")

if torch.cuda.is_available():
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    print(f"현재 사용 중인 GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("GPU를 찾을 수 없어 CPU 모드로 동작합니다. 그래픽 드라이버나 CUDA 설치를 확인하세요.")

# 2. 모델 및 데이터를 GPU로 보내기 (활성화)
# 임의의 텐서(데이터)와 선형 레이어(모델)를 생성합니다.
data = torch.randn(3, 3)
model = torch.nn.Linear(3, 3)

# .to(device) 명령을 통해 데이터를 GPU 메모리로 이동시킵니다.
data = data.to(device)
model = model.to(device)

# 이제 모든 연산은 GPU 안에서 일어납니다.
output = model(data)
print("\n[연산 완료] 결과 데이터 위치:", output.device)
```

핵심은 `torch.device(...)`로 장치를 먼저 결정하고, `.to(device)`로 데이터와 모델을 그 장치로 옮긴 뒤 연산한다는 흐름이다. CUDA가 없으면 자동으로 `cpu`가 선택되므로 코드는 그대로 두고 환경만 갖추면 GPU 연산으로 전환된다.

### 코드분석

`pytorch_check.py`는 GPU(CUDA) 인식 여부를 확인한 뒤, 텐서와 모델을 실제로 GPU로 옮겨 연산까지 성공하는지 검증하는 코드다. 여기서 정한 `device` 지정 패턴은 이후 `cnn_pytorch.py`를 포함한 모든 PyTorch 코드에서 그대로 재사용된다.

<details>
<summary>핵심 요약 (블록별 상세 분석)</summary>

**1. 장치 결정**

```python
# GPU(CUDA)가 있으면 'cuda'를 사용하고, 없으면 'cpu'를 자동으로 선택하는 코드입니다.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

`torch.cuda.is_available()`이 `True`면 `"cuda"`, `False`면 `"cpu"`를 선택한다. 이 한 줄이 이후 모든 텐서·모델의 연산 장치를 결정하는 기준점이 된다.

**2. 인식 결과 출력**

```python
if torch.cuda.is_available():
    print(f"사용 가능한 GPU 개수: {torch.cuda.device_count()}")
    print(f"현재 사용 중인 GPU 이름: {torch.cuda.get_device_name(0)}")
else:
    print("GPU를 찾을 수 없어 CPU 모드로 동작합니다. 그래픽 드라이버나 CUDA 설치를 확인하세요.")
```

`device_count()`와 `get_device_name(0)`은 GPU가 있을 때만 의미가 있으므로 `if` 분기 안에서만 호출한다. GPU가 없을 때 바로 호출하면 오류가 나므로 분기 순서를 지켜야 한다.

**3. 텐서·모델 생성과 장치 이동**

```python
# 임의의 텐서(데이터)와 선형 레이어(모델)를 생성합니다.
data = torch.randn(3, 3)
model = torch.nn.Linear(3, 3)

# .to(device) 명령을 통해 데이터를 GPU 메모리로 이동시킵니다.
data = data.to(device)
model = model.to(device)
```

기본적으로는 CPU에 생성되므로 `.to(device)`를 호출해야만 실제로 GPU 메모리로 옮겨진다. 데이터와 모델을 각각 이동시켜야 하며, 둘 중 하나라도 빠지면 장치 불일치 오류가 발생한다.

**4. 연산 및 결과 확인**

```python
output = model(data)
print("\n[연산 완료] 결과 데이터 위치:", output.device)
```

`model(data)`는 선형 변환(`y = xW^T + b`)을 수행한다. `output.device`를 출력해 연산 결과가 실제로 원하는 장치에서 만들어졌는지 최종 확인한다.

관련 코드: 이 코드의 `device` 결정·이동 패턴은 `cnn_pytorch.py`에서 모델 전체와 매 배치의 이미지·레이블에 그대로 적용된다.

</details>


## TensorFlow 환경 — 자동으로 GPU를 우선 사용하는 방식

TensorFlow는 PyTorch와 달리 GPU가 존재하면 자동으로 우선권을 주어 GPU에서 연산을 처리한다. 따라서 장치를 명시적으로 지정하지 않아도 되지만, 인식이 잘 되었는지 확인하는 절차는 반드시 필요하다.

```
python -c "import tensorflow as tf; 
print('인식된 GPU 목록:', tf.config.list_physical_devices('GPU'))"
```
구글 코랩
```
!python -c "import tensorflow as tf; 
print('인식된 GPU 목록:', tf.config.list_physical_devices('GPU'))"
```
실행 결과
```
인식된 GPU 목록: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```


### `03_소스코드/tensorflow_check.py`

```python
import tensorflow as tf

# 1. 시스템에 등록된 GPU 장치 목록 확인
gpu_devices = tf.config.list_physical_devices('GPU')

print(f"인식된 GPU 개수: {len(gpu_devices)}")

if gpu_devices:
    for i, gpu in enumerate(gpu_devices):
        print(f"GPU 장치 [{i}]: {gpu.name}")
    print("\nTensorFlow는 연산 시 이 GPU를 자동으로 사용(활성화)합니다.")
else:
    print("\n주의: GPU 장치가 인식되지 않았습니다. CPU로만 연산이 진행됩니다.")

# 2. 명시적으로 특정 장치를 지정하여 연산하고 싶을 때 (예시)
# 기본적으로 자동 지정되지만, 아래와 같이 강제 지정할 수도 있습니다.
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("\n[연산 완료] GPU:0 장치에서 행렬 곱셈 성공")
except RuntimeError as e:
    print(e)
```
결과 확인
```
인식된 GPU 개수: 1
GPU 장치 [0]: /physical_device:GPU:0

TensorFlow는 연산 시 이 GPU를 자동으로 사용(활성화)합니다.

[연산 완료] GPU:0 장치에서 행렬 곱셈 성공
```

`tf.device('/GPU:0')`처럼 특정 장치를 강제 지정하는 구문도 함께 익혀두면, 여러 GPU가 있는 환경에서 연산을 분산시킬 때 활용할 수 있다.

### 코드분석

`tensorflow_check.py`는 시스템에 등록된 GPU를 인식하는지 확인하고, 특정 GPU를 명시적으로 지정해 연산해보는 코드다. TensorFlow는 GPU가 있으면 자동으로 우선 사용하므로, `pytorch_check.py`처럼 매번 `device`를 지정하는 코드가 없다는 점이 가장 큰 차이다.

<details>
<summary>핵심 요약 (블록별 상세 분석)</summary>

**1. GPU 목록 조회**

```python
gpu_devices = tf.config.list_physical_devices('GPU')

print(f"인식된 GPU 개수: {len(gpu_devices)}")

if gpu_devices:
    for i, gpu in enumerate(gpu_devices):
        print(f"GPU 장치 [{i}]: {gpu.name}")
    print("\nTensorFlow는 연산 시 이 GPU를 자동으로 사용(활성화)합니다.")
else:
    print("\n주의: GPU 장치가 인식되지 않았습니다. CPU로만 연산이 진행됩니다.")
```

`tf.config.list_physical_devices('GPU')`는 물리적으로 인식된 GPU 장치의 리스트를 반환한다. PyTorch의 `torch.cuda.is_available()`처럼 별도의 불리언 확인 함수 없이, 리스트의 존재 여부 자체로 GPU 유무를 판단한다.

**2. 특정 장치 강제 지정**

```python
try:
    with tf.device('/GPU:0'):
        a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
        b = tf.constant([[5.0, 6.0], [7.0, 8.0]])
        c = tf.matmul(a, b)
        print("\n[연산 완료] GPU:0 장치에서 행렬 곱셈 성공")
except RuntimeError as e:
    print(e)
```

`with tf.device('/GPU:0'):` 블록 안의 텐서와 연산은 0번 GPU에서 실행되도록 강제한다. GPU가 없거나 지정한 장치를 쓸 수 없으면 `RuntimeError`가 발생할 수 있으므로 `try/except`로 감싸 프로그램이 중단되지 않도록 방어한다.

비교: pytorch_check.py는 `device` 변수를 만들어 `.to(device)`로 매번 명시적으로 이동시키는 반면, 이 코드는 기본적으로 아무 지정 없이도 TensorFlow가 알아서 GPU를 쓰고, 특정 장치를 강제하고 싶을 때만 `with tf.device(...)`를 쓴다.

</details>

## PyTorch와 TensorFlow의 GPU 처리 차이 — 핵심 정리

| 항목            | PyTorch                                       | TensorFlow                               |
| ------------- | --------------------------------------------- | ---------------------------------------- |
| GPU 사용 방식     | `torch.device(...)`로 직접 지정, `.to(device)`로 이동 | GPU가 있으면 자동으로 우선 사용                      |
| 인식 확인 명령      | `torch.cuda.is_available()`                   | `tf.config.list_physical_devices('GPU')` |
| 특정 장치 강제 지정   | `.to("cuda:0")`                               | `with tf.device('/GPU:0'):`              |
| CUDA 미인식 시 동작 | 코드 수정 없이 자동으로 `cpu` 선택                        | CPU로만 연산, 경고 없이 진행될 수 있음                 |

## 장치가 잡히지 않을 때 체크리스트

그래픽카드가 분명히 있는데도 `False`가 뜨거나 장치 개수가 0으로 나온다면 아래 원인 중 하나에 해당한다.

- NVIDIA 드라이버가 설치되지 않았거나 최신 버전이 아니다.
- CUDA Toolkit과 cuDNN 라이브러리가 설치된 Python / PyTorch / TensorFlow 버전과 호환되지 않는다.
- Windows/Linux 환경에서 CPU 전용(CPU-only) 버전의 PyTorch/TensorFlow 패키지를 잘못 설치했다. 이 경우 패키지를 지우고 공식 홈페이지의 CUDA 전용 명령어로 재설치해야 한다.

## 다음 장과의 연결

이 장에서 확인한 GPU 인식 여부는 2장 CNN 모델 학습 속도에 직접 영향을 준다. <br>
실습 전에 반드시 `torch.cuda.is_available()` 결과를 먼저 점검하도록 안내한다.
