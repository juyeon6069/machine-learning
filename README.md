## 0: Math_prelim

- Tensor: 딥러닝 연산을 위한 숫자의 다차원 컨테이너
    - **왜 머신러닝에서 Tensor를 쓰나?**
        - 모델의 입력·출력·가중치는 모두 다차원 데이터
        - GPU(행렬 연산에 특화)로 빠르게 계산 가능
        - PyTorch나 TensorFlow의 기본 자료형이 Tensor
    
    | 차원 | 이름 | 예시 |
    | --- | --- | --- |
    | 0D | 스칼라 | `3`, `0.7` |
    | 1D | 벡터 | `[1, 2, 3]` |
    | 2D | 행렬 | `[[1,2],[3,4]]` |
    | 3D 이상 | 텐서 | 이미지 (H×W×C), 배치 데이터 |
  
- Optimization
    - 머신러닝 모델은 **손실 함수(loss)를 최소화하도록 파라미터를 조정**하는 문제.
        
        예:
        
        - 선형회귀: MSE 최소화 → 정규분포 기반 MLE
        - 로지스틱 회귀: Cross-Entropy 최소화 → 베르누이 분포 기반 MLE
        - 신경망: 다양한 복합적 손실 최소화 → Softmax 분포 기반 MLE
        
        최적화 문제의 기본 형태는: 
        
        여기서
        
        - `θ` : 모델 파라미터
        - `L(θ)` : 손실 함수
      
        
        **최적화에서 주로 하는 것**
        
        - Loss가 줄어드는 방향으로 파라미터를 업데이트
        - 대부분 경사하강법(gradient descent) 기반
    - Maximize/minimize an objective function(최적의 해를 최소화/최대화로 찾기 위한 함수) with respect to a set of constraints
    - Gradient descent: 1차 반복 최적화 알고리즘

## 1: Maximum Likelihood Estimation

### 1) Maximum Likelihood:

- 파라미터를 추정하는 *방법론*
- 목적: 주어진 데이터를 가장 잘 설명하는 파라미터를 찾는 것
- MLE는 다음을 최대로 만드는 파라미터 θ 를 찾는 것:


### 2) Bayesian parameter estimation

- **MLE는 데이터로만 파라미터를 추정** → 하지만 데이터가 부족하면 불안정함
- 베이지안 관점에서는 **파라미터도 확률을 가진다고 보고 추정**함.
    - 사용시점
        - **데이터 부족**
        - **파라미터의 신뢰도(uncertainty)**
        - **사전 지식(prior)을 반영해야 할 때**
    - 사전확률(prior): 데이터 보기 전 파라미터에 대한 믿음
    - 사후확률(posterior): 데이터를 본 후 업데이트된 믿음
    - 즉, **새로운 데이터가 들어오면 prior가 업데이트되며 정교한 posterior가 만들어진다.**
- MAP(maximum a posteriori): MLE를 안정화한 버전 (MAP ⊂ Bayesian Estimation)
    - MLE는 “데이터만 믿는 방식”. → 데이터 부족/노이즈 문제 존재(과적합 발생)
    - 그래서 prior(사전지식)을 넣어서 안정화한 것이 MAP (MAP=MLE+정규화)

| 방식 | 개념 | 직관 |
| --- | --- | --- |
| MLE | 데이터만 보고 파라미터 추정 | 현재 데이터만 믿음 |
| Bayesian | prior + 데이터로 추정 | 기존 믿음 + 데이터 둘 다 반영하자 |

## 2. 선형모델: Regression

### 1) Linear Regression

- 입력 X로부터 **연속적인 값 Y**(숫자)를 예측하는 모델
- MLE → 정규분포라고 가정한 likelihood(주어진 관측값이 특정 확률분포로부터 나왔을 확률)를 최대화하는 것
- Loss: MSE
- R-squared (결정계수): 모델이 전체 변동성 중 얼마나 설명했는가(1에 가까울수록 좋은 모델)

### 2) Logistic Regression

- 입력 X로부터 **확률/이진 분류**를 예측하는 모델.
    - MLE → 베르누이 확률을 최대화하는 것
    - Loss: Cross-Entropy
- 둘 다 같은 흐름에서 돌아감:
    1. 모델을 세운다 (parameterizing)
    2. 예측한다
    3. loss 계산한다
    4. optimizer로 θ 업데이트
    5. bias-variance 고려해서 규제(MAP) 적용

## 3. 비선형모델

### 1) KNN

- 새로운 데이터가 들어오면 가장 가까운 K개 데이터를 찾아 그들의 다수결(분류) 혹은 평균(회귀)로 예측하는 알고리즘
    
    → Lazy learning(별도 학습 과정이 없음)
    
- (Decision Boundary) K = 1 → 경계가 매우 들쭉날쭉 (overfitting) , K=15 → 경계가 부드러움 (underfitting)
- K값 → bias-variance 조절
- Weighted KNN: 가까운 이웃이 더 중요하다는 아이디어(경계 근처의 데이터에 민감하게 반응 → 더 안정적)
- 사용시점
    - 모델이 매우 간단하고 직관적일 때
    - 데이터가 적고, 차원이 낮고, 패턴이 비교적 단순할 때
    - 설명력이 좋아서 baseline 모델로 사용
- 문제점
    - **Scale-sensitive**
        - “가까운 거리”가 핵심인 알고리즘이라, **특성의 단위(scale)가 다르면 성능 크게 망가짐**
            -> 그래서 **표준화(Standardization) 필수**
            
    - **High computational cost**
        - 예측할 때마다 전체 데이터를 훑어서 **모든 샘플과 거리 계산**
        - 데이터가 커지면 → **느림**
    - **Curse of dimensionality**
        - 차원이 높아지면 데이터가 희소(sparse)해져서 → “가까운 이웃”이라는 개념이 흐려짐
        - 고차원에서는 KNN 성능이 급격히 나빠짐

### 2) Decision tree

- 데이터를 분할(split)하는 질문들의 트리를 만들어 예측
- entropy: ‘얼마나 섞여있는가’를 측정하는 지표
    - 결정트리는 **혼잡도(disorder)가 낮아지는 방향으로 분할**하려 함.
    - Threshold on entropy: split 했을 때 entropy가 충분히 감소하지 않으면 중단 → 과적합 방지
- (Decision boundary) 트리는 **축을 따라 사각형으로 쪼개는 경계(axis-aligned split)**
- Node split criteria: 노드를 나누었을 때, purity가 높아지는 split을 선택
- Cross-validation: tree depth, min_samples_leaf 등 hyperparameter 최적화 시 사용(max depth, min samples leaf 등을 튜닝)
- 사용시점
    - **설명 가능한 모델**이 필요할 때
    - 비선형 분류/회귀가 필요할 때
    - feature preprocessing 필요 없음(스케일 영향 없음, 결측치 처리 강함)

## 4. Model Selection

### 1) Model evaluation

- 요약
    
    모델 선택은 validation 기반으로 hyperparameter(λ 포함)를 선택하는 과정
    
- Model Selection: 새로운 데이터(=test)에서도 잘 일반화되는 모델을 고르는 것
    - 모델 선택 = validation 기반 + cross-validation 기반 하이퍼파라미터 튜닝
- Validation: Internal / External
- Cross-validation: 데이터가 적을 때 데이터 전체를 효율적으로 쓰기 위한 방법
    - **LOOCV (Leave-One-Out CV)**
        - 샘플이 n개면 **n번 학습**
        - 매번 1개를 검증, 나머지 n−1개를 학습
        - 장단점
            - 장점: 데이터 최대 활용
            - 단점: 너무 느림, variance가 높을 수 있음
    - **K-fold Cross-validation**
        - 데이터를 K개 fold로 나눔 (K=5 or 10이 일반적)
        - 각 fold를 1번씩 validation으로 사용
        - K번 학습
        - 장단점
            - 장점: 안정적, 효율적
            - 단점: 여전히 K번 학습해야 하므로 비용 큼
- CV-based Hyperparameter Selection: Cross-validation으로 hyperparameter를 선택하는 방식.
    - for each hyperparameter value h:
             K-fold cross validation 실시
             (평균 validation 성능 계산)
    성능 가장 좋은 h 선택
- Fine-tuning using validation set
    - train set → CV → candidate model 선택
    validation set → fine-tuning → best model 선택
    test set → 최종 평가

### 2) Regularization

- 흐름
    
    정규화는 목적함수에 패널티를 추가하는 과정이며, 
    
    λ가 그 패널티의 강도를 조절하고, 
    
    L1 등 일부 정규화는 희소성을 만들어내며, 
    
    이 모든 과정은 Bayesian prior를 넣은 것과 동일하다.
    
- 목적: Validation 성능 최적화 위해 머신러닝 모델이  너무 복잡해지는 것(=overfitting)을 막기 위해 추가적인 패널티를 주는 기법.
- Regularized Objective (정규화된 목적 함수)
    - 정규화는 실제로 **목적 함수(=loss)** 안에 다음 항을 추가하는 방식으로 구현됨
    - Regularized Loss(Objective) = 데이터 오차(loss) + λ * Regularization term
- Regularization Parameter λ (람다: 패널티를 얼마나 줄지)
    - λ ↑ → Regularization 더 강함 → 모델이 간단해짐(Bias↑, variance↓)
    - λ ↓ → Regularization 약함 → 모델이 데이터에 더 맞춤(Variance↑, bias↓)
- Sparsity-inducing Regularization (희소성 유도 정규화)
    - 필요 없는 파라미터를 죽이는 정규화
        - 파라미터를 0으로 만들어 **특성 선택(feature selection)** 효과
        - 모델 단순, 해석성 증가
    - 대표적으로 L1(Lasso)
- Bayesian Point of View (정규화를 Bayesian 관점에서 해석하는 방법)
    - 정규화 = 파라미터에 대한 prior(사전 신념)를 추가하는 Bayesian 방식과 동일.
 
## 5. NN
### 1) Forward Propagation

- 입력 → 가중치 연산 → 비선형 활성화 → 다음 레이어 전달(모델이 prediction을 만드는 과정)
- 목적:
    - 입력 데이터를 레이어를 거쳐 변환하여 **output(예측)** 생성
    - Backpropagation을 위해 중간 값(z, a)을 모두 저장(gradient 계산 용도)
- 신경망의 각 레이어는 아래 연산을 수행
    
    **z = W·x + b**
    
    **a = activation(z)**
    
- Activation을 통과해야 **비선형성 확보 → XOR 같은 문제 해결 가능**
- Forward는 계산 그래프를 따라 한 방향으로 흐름

### 2) Backward Propagation

- 모델이 한 번 예측한 후 → loss 계산
- 그 뒤 loss를 줄이기 위해 **각 레이어의 gradient를 계산**하고, 가중치를 업데이트하는 과정.
- 목적: Loss를 줄이기 위해 **각 weight가 얼마나 바뀌어야 하는지**를 자동으로 계산하는 알고리즘
- 단계:
    - (1) **Loss 미분 (∂L/∂output):** 예측값이 얼마나 틀렸는지를 계산
    - (2) **Chain rule 기반 gradient 전파**
        - 출력 → 마지막 레이어 → 중간층 → 입력 방향으로 거꾸로 미분
        - 연쇄 법칙(Chain Rule)을 적용하여 미분값 전달
    - (3) **Weight 업데이트**
        - Gradient Descent: **W := W - η * ∂L/∂W**

### 3) RELU

- 딥러닝에서 가장 널리 쓰이는 activation function.
- a = max(0, z)
- 중요한 이유
    - Sigmoid / Tanh의 문제
        - 깊은 신경망에서 **Vanishing Gradient** 발생
        - gradient가 0에 가까워짐(for sigmoid: 0~1 구간)
    - ReLU의 장점
        - gradient가 1 또는 0 → 사라지지 않음
        - 깊은 네트워크에서도 잘 학습됨
        - 계산 매우 빠름 (max 연산)
    - **ReLU 사용 시 주의점**
        - Dead ReLU 문제: 입력이 항상 0 이하이면 gradient = 0 → neuron이 죽음 ⇒ Leaky ReLU 등으로 해결 가능

### 4) **신경망 설계 요소**

- **Weight Initialization (가중치 초기화)**
    - **왜 중요한가?**
        - Vanishing & Exploding Gradient
            - 너무 큰 초기값 → gradient 폭발
            - 너무 작은 초기값 → gradient 소실
            - ⇒ 역전파(backpropagation) 과정에서 gradient가 레이어를 거치며 점점 작아지거나 커짐
        - 잘못 초기화하면 학습이 시작도 안 됨
    - **대표 방식**
        - **Xavier initialization (tanh 계열)** : 각 레이어의 출력 분포가 입력 분포와 비슷하도록 초기 가중치를 조절
        - He initialization (ReLU 계열)
        
        → 활성화 함수와 맞춰야 “분산이 잘 유지됨(variance-preserving)”
        
- **Dropout:** 학습 시 무작위로 뉴런을 일정 비율 끄는 기술.
    - **목적**
        - 신경망이 특정 뉴런이나 특징에만 의존하는 현상(Overfitting) 방지
        - 네트워크가 특정 weight에 과도하게 의존하는 것을 막음
        - 결국 **앙상블 효과**를 만듦
- **Ensemble:** 여러 모델을 결합해서 더 좋은 성능.
    - Dropout 자체가 mini ensemble 효과
    - 서로 다른 초기값으로 모델 여러 개 학습 → 평균 / 투표
- Optimizer(최적화)
    - SGD
    - Adam
- **Network Architecture 설계 전략**
    - **(1) Fast Forward**
        - Residual connection (Skip connection)
        - Deep residual network(ResNet)의 핵심
        - "입력 x를 몇 개 레이어 뒤로 바로 전달"
        
        → Vanishing Gradient 해결
        
        → 깊은 네트워크 가능
        
    - **(2) Split & Merge**
        - Inception 구조 (GoogLeNet)
        - 여러 가지 filter(1x1, 3x3, 5x5)를 병렬로 적용해 feature 다양성 확보
    - **(3) RNN (Recurrent Neural Network)**
        - 시계열/자연어 처리 등을 위해 등장한 구조.
        - 특징
            - 이전 시점 hidden을 다음 입력과 함께 사용
            - sequence 데이터 처리 가능
        - 단점
            - Gradient vanishing 높음
                
                → LSTM, GRU 개발됨
                


## 6. ConvNet (Convolutional Neural Network)

### 1) ConvNet의 기본 흐름

CNN은 주로 이미지 데이터를 다룰 때 사용, 일반적인 Fully Connected Network(FNN)와 달리 **공간적 구조(spatial structure)**를 활용

- **입력 이미지** → **Convolution + ReLU** → **Pooling** → 반복 → **Fully Connected Layer** → **출력**

이 흐름을 통해 **이미지 특징(feature)**를 단계적으로 추출하고 최종적으로 분류(classification)나 예측(prediction)을 수행

### 2) Convolution Layer (합성곱 레이어)

- **목적**: 이미지에서 특징(feature) 추출
- **핵심 개념**
    - 필터(Filter, 커널, Kernel): 작은 행렬, 예: 3×3, 5×5
    - **Stride**: 필터 이동 간격
    - **Padding**: 경계 처리, SAME/VALID
    - CNN의 핵심은 **지역적 패턴 인식(Local pattern)**:
        - 모서리(edge)
        - 선(line)
        - 숫자 형태 패턴(feature)
- **출력 크기 계산**:
Output size = (N - F) / stride + 1

- NNN: 입력 크기
- FFF: 필터 크기
- **활용 예**: edge detection, pattern extraction 등

### 3) ReLU (Rectified Linear Unit)

- **목적**: 비선형성 부여, 학습 속도 향상
- **정의**:
    f(x)=max⁡(0,x)

    
- Convolution 후 적용: **conv → ReLU → conv → ReLU → …**

### 4) Pooling Layer (Subsampling)

- **목적**: 차원 축소, 연산량 감소, translation invariance 확보
- **주요 방식**
    - **Max pooling**: 영역 내 최대값
    - **Average pooling**: 영역 내 평균값
- 보통 **2×2 영역, stride=2**를 많이 사용

### 5) Flatten + Fully Connected Layer

- CNN 마지막 단계에서 이미지 특징을 **1차원 벡터로 펼친 후** 연결
    - CNN은 특징을 추출하고 Flatten 이후 FC Layer에서 분류(classification)를 수행함
- **출력**: 분류 문제의 경우 softmax, 회귀 문제

### 6) Full Network 구성 예시

```
conv-relu → conv-relu → pool
conv-relu → conv-relu → pool
flatten → fully connected → softmax
```

- **LeNet-5**: 초기 CNN, 손글씨 숫자 MNIST 분류용
- **AlexNet**: 깊고 큰 CNN, ImageNet 대회 우승
- **GoogLeNet (Inception)**: 병렬 필터 사용, 연산 최적화
- **ResNet**: residual connection, 매우 깊은 네트워크에서도 학습 가능

## 7. RNN(Recurrent Neural Network)

- 순차적 데이터(sequence data)를 처리하는 신경망.
    - 입력의 순서와 이전 정보를 고려하여 출력 예측.
    - Hidden state: 과거 정보를 기억하여 다음 단계에 전달.
- 응용 예시
    - **Vanilla RNN**: 가장 기본 구조. 작은 시퀀스나 간단한 패턴에 사용.
    - **Image Captioning**: 이미지 특징을 추출한 후 시퀀스 단어를 생성.
    - **Sentiment Classification**: 문장의 감정을 순서대로 읽어 최종 감정을 예측.
    - **Machine Translation**: 입력 문장을 순서대로 처리하여 다른 언어 문장 생성.
    - **Video Classification**: 프레임별 특징을 순서대로 처리해 동영상 분류.
- Long Sequence RNN
    - 시퀀스가 길어지면 **Vanishing Gradient** 문제가 발생해 장기 의존성(long-term dependency)을 학습하기 어려움
    - 해결방법
        - **LSTM(Long Short-Term Memory)**
            - Gate 구조: Forget, Input, Output
            - Cell state 유지 → 장기 기억 가능.
        - **GRU(Gated Recurrent Unit)**
- Stacked RNN + Softmax Layer
    - RNN을 층층이 쌓아 **Stacked RNN**을 만들 수 있음 → 더 복잡한 시퀀스 패턴 학습 가능.
    - 마지막 출력층에 Softmax를 붙여 분류 문제 처리.
- Dynamic RNN
    - 입력 시퀀스 길이가 가변적일 때 사용.
    - TensorFlow 같은 프레임워크에서는 `dynamic_rnn`으로 처리하여 메모리와 계산 효율을 높임.
- RNN with Time Series Data
    - 주가, 날씨, 센서 데이터 등 **연속적 시간 데이터** 분석에 RNN이 널리 사용됨.
    - 과거 데이터를 기반으로 미래를 예측할 수 있음.
