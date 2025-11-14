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
