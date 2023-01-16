# ml-project-template

## Overview
일반적으로 머신러닝을 입문하면서 공부할때는 Jupyter를 이용하여 학습을 하지만, 효율적인 연구 및 개발 환경을 갖추기 위해서는 IDE (pycharm)를 이용하여 체계적인 코드 구조를 갖추어야 한다. 물론 Jupyter로도 충분히 연구가 가능하지만 다음과 같이 실질적인 활용에 제약이 있다. 

- 대규모 데이터에 대한 실험을 수행할 때는 서버에서 백그라운드로 실행하고 실험 수행 도중 다른일 (논문 작성 등)하게 되는데 Jupyter로 하게 되었을 때 웹 브라우저가 꺼지게 되면 실험 전체가 중단된다 (shell과 몇몇 linux tools을 이용하면 안정적으로 할 수 있음)
- 여러가지 하이퍼파라미터에 대해 다양한 실험을 병렬적으로 해야 하는데 Jupyter로는 이를 자동화 하기 어렵다 (terminal input을 처리할 수 있게 코딩하면 이를 쉽게 할 수 있음).
- 향후 github을 통해 코드 배포를 할 때 Jupyter 기반의 코드를 다시 정리해서 배포용 코딩을 따로 해줘야 한다.

이러한 제약 때문에 보통 Jupyter는 개발 과정에서 개발 모듈의 작동을 interactive하게 파악하기 위해 테스트용으로 활용하거나, 실험 결과의 시각화, 개발한 라이브러리의 교육용 hands-on tutorial을 작성하는데 활용된다 (결과적으로 사용자와의 interaction이 필요할 때 효과적임). 

본 문서에서는 서버 작업을 염두에 둔 머신러닝 프로젝트 개발 과정을 위한 가이드라인을 제공하고자 한다. 연구 수준에서 서버에서 작업을 할 수 있다면 어떠한 코드 구조든 상관 없지만, 이러한 환경을 처음 접해보는 사람의 입장에서 쉽게 접근해볼 수 있도록 MNIST 데이터에 대해 아주 간단한 모델 (linear+softmax)로 예시를 들고자 한다. 

## Installation
* Step 1. git clone https://github.com/jbnu-dslab/ml-project-template
* Step 2. `ml-project-template`에서 사용되는 라이브러리를 설치 (아래 명령어를 수행하기 전에 conda activate를 해줘야함)
```shell
pip install -r requirements.txt
```
* Step 3. PyCharm으로 열기 (이때 conda는 step2에서 activate된 환경으로 설정해줘야 함)
  - PyCharm을 이용한 개발 환경 셋업 방법은 [link](https://www.notion.so/1-Python-development-environment-for-macOS-e8dd60c1efd4435fb43a8fa4877a43c9) 참고





## Project structure
`ml-project-template`는 다음과 같은 구조로 이루어져있다. `ml-project-template` 코드의 세부 내용을 분석할 때는 `main.py`를 시작점으로 해서 살펴보면 된다. 

```shell
├── README.md
├── datasets                        # datasets을 저장하는 폴더 (다운로드 되면 자동으로 생성됨)
│   └── MNIST                       
└── src                             # source codes를 저장하는 폴더
    ├── data.py                     # datasets (DataSet, DataLoader) 관련 작업을 처리하는 script
    ├── experiments                 # 실험 scripts를 저장하는 폴더
    │    ├── __init__.py
    │    └── exp_hyper_param.py
    ├── main.py                     # 사용자 입력을 처리하는 script
    ├── models                      # model의 코드를 저장하는 폴더 (여러 모델을 구현한다고 가정)
    │    ├── __init__.py
    │    └── mymodel                # 'mymodel'의 코드를 저장하는 폴더
    │        ├── __init__.py
    │        ├── model.py           # mymodel 구현 담당 (주로 forward 함수 구현)
    │        ├── train.py           # data, hyper_param을 받아 mymodel 훈련 담당 (주로 gradient descent & backprop)
    │        └── eval.py            # test data에 대해 훈련된 mymodel 평가 담당 (주로 정확도 측정)
    └── utils.py
```

## How to call this in a terminal?
`ml-project-template`는 사용자의 입력을 받아 터미널 상에서 다양하게 실험해볼 수 있는 방법을 제공한다 (main.py에 구현됨). 
터미널에서 사용자 입력을 처리하는 여러가지 파이썬 라이브러리가 있지만 API 사용이 직관적인 [fire](https://github.com/google/python-fire) 를 통해서 구현했다.
먼저 터미널을 열고 프로젝트 루트에서 `src` 디렉토리로 이동한후 다음과 같은 형태로 명령어를 수행할 수 있다. 
```
python -m main --model mymodel
               --seed 777
               --batch_size 100
               --epochs 15
               --learning_rate 0.001
```

`python -m main --help`를 입력하면 각 argument에 대한 설명을 확인 할 수 있다. 아래 메시지는 main.py의 main() 함수에서 주석에 해당되는 내용이 파싱되어서 출력되는 것이다. 
```
NAME
    main.py - Handle user arguments of ml-project-template

SYNOPSIS
    main.py <flags>

DESCRIPTION
    Handle user arguments of ml-project-template

FLAGS
    --model=MODEL
        Default: 'mymodel'
        name of model to be trained and tested
    --seed=SEED
        Default: -1
        random_seed (if -1, a default seed is used)
    --batch_size=BATCH_SIZE
        Default: 100
        size of batch
    --epochs=EPOCHS
        Default: 15
        number of training epochs
    --learning_rate=LEARNING_RATE
        Default: 0.001
        learning rate
```

위와 같이 terminal 상에서 쉘 명령어 처리가 가능하면 다음과 같은 활용이 가능해진다. 
* 여러가지 파라미터에 대해 병렬적인 실험이 쉬워진다 (CPU 가동가능한 범위내에서 쉘 명령어를 여러번 background 호출하면 운영체제에 의해 자동적으로 스케쥴링되면서 병렬수행됨)
  - 실험 결과등은 디스크에 따로 저장해두는 방식으로 해야 한다 (결과를 터미널 상에서 눈으로 보기 어려워짐)
  - 당연하게도 너무 많은 프로세스 호출이 일어나면 scheduling overhead로 느려진다 (그래서 동시에 수행되는 명령어수를 잘 제어해야한다)
* `tmux`는 linux에서 하나의 화면에서 여러 개의 터미널을 띄우고 개별적으로 수행가능하게 해주는 도구인데, tmux에서 session은 사용자가 끄거나 서버가 다운 되지 않는 이상 계속 유지된다. 그래서 서버에 접속한 뒤 tmux를 이용하면 background 호출을 위한 shell script를 작성하지 않고도 안정적으로 실험을 수행할 수 있다 (이런게 필요한 이유는 client의 터미널을 종료하면 자동으로 서버 연결이 끊기기 때문이다. tmux는 서버에서 돌기 때문에 client 연결이 끊겨도 서버에서는 계속 수행된다).
  - 보통 급하게 실험을 해야 하거나 shell scripting을 하는게 귀찮을 때 tmux를 많이 활용한다. 

이외에도 일반적으로 머신러닝 프로젝트를 github에서 관리할 때 사용자 입력을 처리할 수 있도록 해주기 때문에 (사용자가 받아서 바로 써보게 하기 위해) 사용자의 입력을 처리하는 방식을 익히면 여러 도움이 된다.  

## How to add a new model?
`models` 폴더에 `mymodel`과 같은 형태로 새로 추가하면 된다. model를 구현하는 데 있어 model, train, eval로 나눈 이유는 역할 단위를 구분하여 각각의 역할을 집중적으로 구현하고 관리를 쉽게 하기 위함이다 (물론 프로젝트 규모가 작으면 이렇게 나눌 필요가 없지만, 여러 모델의 코드를 관리하고 코드가 조금만 길어져도 관리가 힘들어진다). 

## Todo list
* [ ] Write 'how to open this using pycharm?'
* [ ] Write 'how to make an experimental script?'
