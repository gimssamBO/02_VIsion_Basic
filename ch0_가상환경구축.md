# 0장 파이썬 가상환경 구축

- 실습 환경: Windows, Python 3.12, VS Code

> Python 3.12와 VS Code를 설치하고, `conda`로 프로젝트 전용 가상환경을 만든 뒤 장별로 필요한 라이브러리를 설치한다. 가상환경을 쓰면 프로젝트마다 독립된 패키지 버전을 유지할 수 있어, 이후 장에서 다룰 TensorFlow · PyTorch · OpenCV · YOLO 라이브러리가 서로 충돌하지 않는다.

### 개발 환경 준비

- Python 3.12를 다운로드하여 설치한다.
- VS Code를 설치하고 터미널을 연다.

### 가상환경 생성

```
D:\01_AI_Study_KTF_OpenCV\vision>conda create -n dev python=3.12
D:\01_AI_Study_KTF_OpenCV\vision>conda info --envs
D:\01_AI_Study_KTF_OpenCV\vision>conda activate dev
(dev)D:\01_AI_Study_KTF_OpenCV\vision>
```

`conda` 모듈로 `dev`라는 이름의 가상환경을 만든다. 프롬프트 앞에 `(dev)`가 표시되면 가상환경이 활성화된 상태이며, 이후 설치하는 모든 라이브러리는 이 가상환경 안에만 설치된다.

### 실습 라이브러리 설치

장별로 필요한 라이브러리가 다르므로, 아래 표를 기준으로 순서대로 설치한다.

| 장             | 용도                         | 설치 명령                                      |
| ------------- | -------------------------- | ------------------------------------------ |
| 2장 (CNN)      | 텐서플로우 및 수치 연산              | `pip install tensorflow numpy`             |
| 2장 (CNN)      | 파이토치 및 비전 라이브러리            | `pip install torch torchvision torchaudio` |
| 3~4장 (OpenCV) | 이미지 및 카메라 처리               | `pip install opencv-python`                |
| 5장 (YOLO)     | 객체 탐지(PyTorch 등 의존성 자동 포함) | `pip install ultralytics`                  |

requirements.txt = 한 번에 라이브러리 설치하는 텍스트 파일
```txt
# Deep Learning & Computer Vision Environment Requirements
numpy
tensorflow
torch
torchvision
torchaudio
opencv-python
ultralytics
```

```bash
D:\01_AI_Study_KTF_OpenCV\vision>conda install pip
D:\01_AI_Study_KTF_OpenCV\vision>pip install -r requirements.txt
```

설치가 끝나면 각 명령의 마지막 줄에 `Successfully installed ...` 메시지가 출력되는지 확인한다. 예를 들어 텐서플로우 설치 결과는 다음과 같다.

```
Successfully installed absl-py-2.5.0 astunparse-1.6.3 certifi-2026.7.22 charset_normalizer-3.5.0
flatbuffers-25.12.19 gast-0.7.0 google_pasta-0.2.0 grpcio-1.83.0 h5py-3.14.0 idna-3.18
keras-3.15.1 libclang-18.1.1 markdown-it-py-4.2.0 mdurl-0.1.2 ml_dtypes-0.6.0 namex-0.1.0
numpy-2.5.2 opt_einsum-3.4.0 optree-0.19.1 packaging-26.3 protobuf-7.35.1 pygments-2.20.0
requests-2.34.2 rich-15.0.0 setuptools-84.0.0 six-1.17.0 tensorflow-2.21.0 termcolor-3.3.0
typing_extensions-4.16.0 urllib3-2.7.0 wheel-0.48.0 wrapt-2.3.0
```

torch, opencv-python, ultralytics 설치도 같은 방식으로 진행한다. 이미 설치된 공통 의존성(numpy, requests 등)은 `Requirement already satisfied`로 표시되고 다시 설치되지 않는다.

## 다음 장과의 연결

가상환경 구축이 끝나면, 이어지는 1장에서 이 환경 안에 설치된 PyTorch · TensorFlow가 GPU(CUDA)를 실제로 인식하는지 확인한다.
