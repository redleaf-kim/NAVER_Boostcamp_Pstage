## ♻️ 재활용 품목 분류를 위한 Semantic Segmentation

###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 Team Project로 진행됐습니다.
<br>


----
### 🍀  최종 결과 
- [[Semantic Segmentation]](http://boostcamp.stages.ai/competitions/28/overview/description)
    - **1등 (총 21팀)**
    - public  LB : 0.7205 
    - private LB : 0.7043
    - [[발표 자료]](https://www.notion.so/MultiHead_Ensemble-a6d4e3db725a4588ab18ab7ea2551c92#0ace36d4004d4f17913cc543888fa0bd)
    - [[Code]](https://github.com/bcaitech1/p3-ims-obd-multihead_ensemble/blob/master/Hongyeob-Kim/Semantic_Segmentation/)
<br></br>

### 🗄 폴더 구조
```
└── Semantic_Segmentation
     ├── experiments
     ├── src
     ∣    ├── configs
     ∣    ├── models
     ∣    ∣    ├── fcn8s.py
     ∣    ∣    └── hrnet_seg.py
     ∣    ├── augmix.py
     ∣    ├── dataset.py
     ∣    ├── losses.py
     ∣    ├── schedulers.py
     ∣    ├── utils.py     
     ∣    └── warping.py
     ∣
     ├── test_scripts
     ├── augmix_train.py
     ├── ensemble_test.py
     ├── pseudo_train.py
     ├── ensemble_test.py
     ├── train_eval.py
     ├── tta_ensemble_test.py
     └── tta_test.py
```
<br></br>

### 💻 소스 코드 설명
- `augmix.py` : SongbaeMix 오리지널 코드
- `losses.py` : semantic segmentation loss 모아놓은 코드 , import module을 통해 불러와서 train시 사용
- `scheduler.py` : cosine annealing with warm starts를 사용
- `utils.py` : train / valid 코드, 데이터셋 구성을 위한 utils함수 정의
- `experiments` : train arguments 관리를 쉽게하기 위해서 script파일로 훈련 수행
- `test_scripts` : 어떤 모델이나 조건으로 테스트했는지 기억하기 위해서 shell script로 테스트 실행
- `train_eval.py` : train dataset만을 학습하는 코드
- `augmix_train.py` : SongbaeMix를 활용해 학습하는 코드
- `pseudo_train.py` : train dataset과 pseudo dataset을 학습할 때 사용
- `ensemble_test.py` : Ensemble모델로 추론하는 코드
- `tta_test.py` : 싱글모델에 TTA를 적용해 추론하는 코드
- `tta_ensemble_test.py` : Ensemble모델에 TTA를 적용해 추론하는 코드