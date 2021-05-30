## 😷 마스크 착용 상태 분류 Image Classification

###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 Team Project로 진행됐습니다.
<br>

## 목차
+ [최정결과](#Result)
+ [폴더구조](#Directory)
+ [소스코드설명](#Code)
+ [기술적시도](#기술적시도)
    + [모델](#모델)
    + [Loss함수](#Loss-함수)
    + [데이터불균형](#데이터-불균형)
    + [잘못된라벨링](#잘못된-라벨링(mislabeled))
    + [추가데이터](#추가-데이터)
    + [기타](#기타)
+ [어려웠던 점 & 반성할점](#어려웠던-점-반성할-점)
+ [좋았던 점 & 배운 점](#좋았던-점-배운-점)

----
### 🍀 최종 결과 <a name = 'Result'></a>
- [[Image Classification]](http://boostcamp.stages.ai/competitions/1/overview/description)
    - **4등 (총 2xx팀)**
    - private LB : 0.7691
<br></br>

### 🗄 폴더 구조 <a name = 'Directory'></a>
```
└── Image_Classification
     ├── experiments
     ├── infer.py
     ├── infer.sh
     ├── logs
     ├── pseudo_train.py
     ├── pseudo_train.sh
     ├── src
     │   ├── configs
     │   ├── dataset.py
     │   ├── earlyStop.py
     │   ├── loss.py
     │   ├── models.py
     │   ├── trainer.py
     │   └── uda_train.py
     └── infer.py
```
<br></br>

### 💻 소스 코드 설명 <a name = 'Code'></a>
- `pseudo_train.py` : pseudo labeling을 활용한 train 코드
- `earlyStop.py` : overfitting을 방지하기 위한 earlyStop 코드
- `loss.py` : 학습용 loss함수 정의
- `models.py` : 모델 정의
- `trainer.py` : train 코드를 간결하게 하기위한 train helper function
- `uda_train.py` : [[Paper]](https://arxiv.org/abs/1904.12848) 해당 논문에서 나오는 UDA기법을 적용해보려고 시도한 코드
<br></br>
---

## [기술적시도]
- [ ] 시도해 봤지만 잘 동작하지 않은 것들
- [x] 시도하고 좋아서 사용한 것들
<br><br>
  
### **모델** 
- Efficientnet b0-b4, NFNet f0-1를 모두 사용해 봤지만, 일정 수준 이하의 validation loss보다 내려가면 오히려 성능이 하락하는 현상이 발생
- 모델의 capacity 보다는 학습 전략에 초첨을 맞춰야 겠다고 느꼈고, 이후 모델은 ``Efficientnet b3모델기반의 모델을 고정적으로 사용``
<br><br>

### **Loss 함수**
- [x] Arcface loss
    - 롯데정보통신에서 주최한 분류대회에서 1등하신 분이 사용했던 ``Arcface loss`` 
    - angular margin penalty를 통해서 클래스 간의 discrepancy를 향상하는 기법
    - 실제로 전통적인 CE를 활용했을 때보다 좀 더 빨리 수렴했음
<br><br>

### **데이터 불균형**
- [ ] WeightedRandomSampler
    - 데이터 불균형을 해결해 보고자 사용해 봤지만 점수 대폭 하락 
    - 적은 비율을 차지하는 60세 이상의 데이터의 경우에 몇 개 없는 것을 반복적으로 자주 학습하게 되어 overfitting 됬을 것으로 생각
- [x] Class Weight
    - class의 weight를 계산해 CE를 계산해 줄때 사용 → ``arcface loss와 함께 사용하여 효과를 봄``
- [x] Focal Loss
    - Object Detection과 같이 극단적인 imbalance가 일어나는 태스크에서 자주 사용함
    - 쉬운 샘플에 대해서는 작은 가중치를 부여하고, 어려운 샘플에 대해서는 큰 가중치를 부여해 학습의 어려운 예제에 집중할 수 있도록 도움
- [x] Age filter
    - 60세 이상의 그룹에 해당하는 나이가 60세 밖에 없었기 때문에 50대 후반의 데이터를 60세 이상으로 바꿔보자는 아이디어에서 시작
    - 피어세션에서의 공유를 통해서 58세가 가장 좋은 결과가 나왔다고 들었지만, ``나는 59세가 가장 효과적이엇음``
<br><br>

### **잘못된 라벨링(mislabeled)**
- [ ] Label Smoothing
    - 분류 문제에서 label 이 1이라고 해서 반드시 label은 1이 아닐 수 있음, 사람이 annotation을 진행하기 때문에 항상 오류가 존재할 가능성이 있기 때문임
    - 이때 발생할 수 있는 mislabeled 데이터의 영향을 줄이고 모델의 일반화를 높이기 위해서 사용하는 기법임
    - label을 하나의 정수로 부여하는 것이 아니라, 다른 label이 될 수도 있는 가능성을 추가해 줌으로 regularization에 도움을 즘
- [x] 직접 고쳐주기
    - 토론글에 공유된 잘못되어 보이는 경우를 manual하게 고쳐줌
<br><br>
          
### **추가 데이터**
- [ ] UDA(Unsupervised Data Augmentation)
    - labeled 되지 않은 데이터를 활용하는 방법으로 해당 데이터를 augmentation을 거치지 않고 제공했을 때와, augmentation을 거친 후에 제공했을 때의 벡터 값이 유사하게 나오도록 유도함으로써 성능을 끌어 올리는 기법임
    - ``supervised loss``: 전통적인 cross entropy loss를 사용함
    - ``unsupervised loss``: kl-divergence loss를 사용
- [x] Pseudo Labeling 
    - labeled 되지 않은 데이터에 가상의(pseudo) 라벨을 붙여서 학습 시에 활용하는 기법임
    - 온라인에서 구한 추가 데이터를 제공 했을 때는 오히려 성능 하락을 보였음
    - test 데이터를 활용하였을 때 점수 상승이 가장 많이 발생
    - ``test 데이터를 활용하는 것이 data leakage라고 생각되는데`` 이를 활용해서 점수를 올리는 것이 과연 유의미 한것인지 판단이 서지 않음
        - test 데이터 말고 추가적으로 unlabeld 데이터가 있어서 ``semi-supervised를 경험해 볼 수 있었다면 더 재밌는 대회가 됬을 거 같음``
<br></br>
### **기타**
- [x] TTA(Test Time Augmentation)
    - 말 그대로 Inference(Test) 과정에서 Augmentation 을 적용한 뒤 예측의 확률을 평균(또는 다른 방법)을 통해 도출하는 기법
    - 과도한 augmentation을 줬을때 오히려 스코어 하락이 있었고,`` horizontal flip만 추가해 주었을 때 가장 성능향상이 높았음`` 
    - ``f1 score 0.04점이 올랐음``
<br></br>
          

## [어려웠던 점 & 반성할 점]
- 아무래도 age가 가장 어려웠던 것 같음
    - 마스크에 의해서 대부분의 얼굴정보가 가려져 있었음
    - 학습데이터 중간중간 30대처럼 보이지만 label은 24살인 mislabeled 데이터가 간간히 있었음
    - age의 구간대가 ~29, 30~59, 60~의 세구간으로 나누어져 있는데 60세 이상의 데이터는 60세 한타깃 밖에 없어서 데이터 불균형이 너무 심하게 존재함
    - 아무래도 50대 후반 57~59세의 데이터들이 60세와의 차이를 찾는 것이 어려웠음
<br></br>
      
### 안경
- 2700명이나 되는 사람 중에서 안경을 쓴 사람이 약 3~4명정도 존재함
- 10보다 작은 n명이라고 가정한다면, 각 사람마다 마스크 쓴 5장, 이상하게 쓴 1장, 안 쓴것 1장으로 마스크를 쓰지 않은경우는 4장에 불과함
- 2700명의 데이터를 모두 세보면 18900장인데 안경을 쓰고 마스크를 쓰지 않은 것은 겨우 n장이됨
    - 최대치로 잡고 10장이 있다고 한다면, 10/18900은 약 0.06% → 너무 적은 케이스로 학습이 잘 안될 것으로 생각됨
- 실제로 안경을 쓴 테스트 데이터에 대해서 마스크를 쓰지 않아도 마스크를 썻다고 판단하는 경우가 있음
<br></br>
      
### 기타
- shell 스크립트를 통해서 훈련과 추론을 하다보니, parameter들을 바꿔주지 않고 종종 진행하여 시간과 제출 횟수만 날린일이 종종 발생했음, 다음 스테이지에서는 좀더 꼼꼼하게 실험을 해볼 수 있는 방법을 고민하고 있음
<br></br>

- 스스로 도전적인 사람이라고 생각했었는데, 예전에 tensorboard에 크게 데었던 적이 있어서 사용하기가 꺼려졌었음, ``귀찮아 or 굳이?`` >> ``해보자``가 되어서 시도해보지 않은 것들이 크게 느껴짐 ➜ 조금더 프론티어적인 마인드로 임해야겠다고 다짐하게 되었음
<br></br>
      
### 공유문화
- 피어세션에서의 공유는 활발하게 진행했지만, 토론 글을 통해서 공유하려고 하는 노력이 부족했음
- 공유하고자 했던 것들이 대부분 이미 다른 캠퍼분들에 의해서 공유된 상황이라 적을 내용이 없었다는 것 정도인데 되돌아 본다면 참으로 구차한 핑계임
- 다음 프로젝트부터는 ``선제적``으로 ``사소한 것이라도`` 혹은 겹치는 내용이 있더라고 ``나만의 관점에서 정리하여 공유``하는 문화에 동참해 봐야겠음
<br></br>


## [좋았던 점 & 배운 점]
### 피어세션과 토론글을 통한 공유문화    
- 다양한 사람과의 피어세션을 통해서 다양한 시각을 접할 수 있었음
- 우물안의 개구리는 바다가 어떤 것인지 모른다는 말이 있듯 편협한 시각에만 갇혀 있다면 결국 도태되는 것은 본인이라는 것을 굉장히 많이 느꼈음
- 어려움을 겪고 있는 다른 캠퍼들을 같이 해결해 나가면서 서로서로가 몰랐던 것들을 채워주는 성장의 경험을 할 수 있었음
    - stage3&4가 6명이상으로 구성 된 팀으로 진행 된다고 했을때 걱정했던 본인이 부끄러웠음
    - Github를 통해서 같이 코드를 공유하면서 협업해보는 경험을 빨리 가지고 싶어졌음!
<br></br>
        
### 독창성
- 나이를 가늠할 수 있는 요소가 마스트로 가려지지 않은 얼굴과 머리카락인데, 이것이 ``뒤죽박죽 섞여 있는 상태에서도 판별할 수 있다면 좋은 성능을 유도할 수 있지 않을까``라고 하는 고민으로 gridshuffle을 시도해본 캠퍼가 있었음 ⇨ 정말 신선한 아이디어였고, 엉뚱한 아이디어임에도 불구하고 성능향상으로 연결되는 것을 보고 ``엉뚱함``도 생각보다 중요한 요소인 것 같다는 생각을 하게됨
<br></br>


### [훈련 & 추론방법]
- Train
    ```
    sh experiments/experiment_version.sh
    ```
    or 
    ```
    sh pseudo_train.sh
    ```

- Infer
    ```
    sh infer.sh
    ```
<br></br>