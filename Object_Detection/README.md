## ♻️ 재활용 품목 분류를 위한 Object Detection

###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 Team Project로 진행됐습니다.
<br>


----
### 🍀  최종 결과 
- [[Object Detection]](http://boostcamp.stages.ai/competitions/35/overview/description)
    - **2등 (총 21팀)**
    - public  LB : 0.6068
    - private LB : 0.5014
<br></br>

### 🗄 폴더 구조
```
└── Object Detection
     ├── scripts
     ├── src
     ∣    ├── configs
     ∣    ├── demo
     ∣    ├── docker
     ∣    ├── docs
     ∣    ├── mmcv_custom
     ∣    ├── mmdet
     ∣    ├── pretrained
     ∣    ∣    └── swin_base_patch4_window12_384_22kto1k.pth
     ∣    ├── requirements
     ∣    ├── resources
     ∣    ├── tests
     ∣    └── tools
     ∣
     └── test_scripts
```
<br></br>

### 💻 소스 코드 설명
- `src` : [[Original repo]](https://github.com/SwinTransformer/Swin-Transformer-Semantic-Segmentation) mmdetection의 레포에 swin transformer를 추가한 폴더
- `scripts` : train arguments 관리를 쉽게하기 위해서 shell script파일로 훈련 수행
- `test_scripts` : test를 위한 shell scripts파일 폴더