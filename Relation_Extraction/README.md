# Pstage 2 ] KLUE

###### 📚 문장 내 개체간 관계 추출 Realtion Extraction
###### 📌 본 프로젝트는 [_*Naver AI Boostcamp*_](https://www.edwith.org/bcaitech1/)에서 진행됐습니다.
<br></br>

## 📋 Table of content
+ [최정결과](#Result)
+ [대회개요](#Overview)
+ [데이터개요](#Data)
+ [문제정의 및 해결방법](#Solution)
+ [폴더구조](#Directory)
+ [소스코드설명](#Code)
<br></br>
<br></br>


## 🍀 최종 결과 <a name = 'Result'></a>
- [[Relation Extraction]](http://boostcamp.stages.ai/competitions/4/overview/description)
    - Final LB (39/135)
        - Acc: 80.1%
<br></br>
<br></br>



## 🔤 대회 개요 <a name = 'Overview'></a>
관계 추출(Relation Extraction)은 문장의 단어(Entity)에 대한 속성과 관계를 예측하는 문제입니다. 관계 추출은 지식 그래프 구축을 위한 핵심 구성 요소로, 구조화된 검색, 감정 분석, 질문 답변하기, 요약과 같은 자연어처리 응용 프로그램에서 중요합니다. 비구조적인 자연어 문장에서 구조적인 triple을 추출해 정보를 요약하고, 중요한 성분을 핵심적으로 파악할 수 있습니다.
요약된 정보를 사용해 QA 시스템 구축과 활용이 가능하며, 이외에도 요약된 언어 정보를 바탕으로 효율적인 시스템 및 서비스 구성이 가능합니다.

이번 대회에서는 문장, 엔티티, 관계에 대한 정보를 통해 ,문장과 엔티티 사이의 관계를 추론하는 모델을 학습시킵니다. 이를 통해 우리의 인공지능 모델이 엔티티들의 속성과 관계를 파악하며 개념을 학습할 수 있습니다. 우리의 model이 정말 언어를 잘 이해하고 있는 지, 평가해 보도록 합니다.
- input: sentence, entity1, entity2 의 정보를 입력으로 사용 합니다.
```
sentence: 오라클(구 썬 마이크로시스템즈)에서 제공하는 자바 가상 머신 말고도 각 운영 체제 개발사가 제공하는 자바 가상 머신 및 오픈소스로 개발된 구형 버전의 온전한 자바 VM도 있으며, GNU의 GCJ나 아파치 소프트웨어 재단(ASF: Apache Software Foundation)의 하모니(Harmony)와 같은 아직은 완전하지 않지만 지속적인 오픈 소스 자바 가상 머신도 존재한다.
entity 1: 썬 마이크로시스템즈
entity 2: 오라클

relation: 단체:별칭
```
- output: relation 42개 classes 중 1개의 class를 예측한 값입니다.
- 위 예시문에서 단체:별칭의 label은 6번(아래 label_type.pkl 참고)이며, 즉 모델이 sentence, entity 1과 entity 2의 정보를 사용해 label 6을 맞추는 분류 문제입니다.
- 평가방법 
    - 모델 제출은 하루 5회로 제한됩니다.
    - 평가는 테스트 데이터셋의 Accuracy 로 평가 합니다. 테스트 데이터셋으로 부터 관계를 예측한 classes를 csv 파일로 변환한 후, 정답과 비교합니다.
<br></br>
<br></br>


## 💾 데이터 개요 <a name = 'Data'></a>
전체 데이터에 대한 통계는 다음과 같습니다. 학습에 사용될 수 있는 데이터는 train.tsv 한 가지 입니다. 주어진 데이터의 범위 내 혹은 사용할 수 있는 외부 데이터를 적극적으로 활용하세요!

- train.tsv: 총 9000개

- test.tsv: 총 1000개 (정답 라벨 blind)

- answer: 정답 라벨 (비공개)
<br></br>
<br></br>


## 📝 문제정의 및 해결방법 <a name = 'Solution'></a>
- 해당 대회에 대한 문제를 어떻게 정의하고, 어떻게 풀어갔는지, 최종적으로는 어떤 솔루션을 사용하였는지에 대해서는 각자의 wrap up report에서 기술하고 있습니다. 
    - [wrapup report](https://maihon.oopy.io/study/boostcamp/p-stage/relation-extration/wrapup-report)    

- 위 report에는 대회를 참가한 후, 개인의 회고도 포함되어있습니다. 
<br></br>
<br></br>


### 🗄 폴더 구조 <a name = 'Directory'></a>
```
└── Relation_Extraction
     ├── aug_data
     ├── data
     └── src
          ├── experiments
          ├── enriching_inference_ensemble.py
          ├── enriching_inference.py
          ├── enriching_kfold.py
          ├── enriching_train.py
          ├── load_data.py
          ├── loss.py     
          └── enriching_model.py
```
<br></br>
<br></br>


### 💻 소스 코드 설명 <a name = 'Code'></a>
- `enriching_inference_ensemble.py` : Ensemble enriching model 추론코드
- `enriching_inference.py` : 단일 enriching model 추론코드
- `enriching_kfold.py` : Enriching model kfold 학습코드
- `enriching_model.py` : [[Paper]](https://arxiv.org/pdf/1905.08284.pdf)에 소개한 Enriching model 코드
- `enriching_model.py` : 단일 Enriching model 학습코드
- `load_data.py` : 데이터셋 로딩코드
- `loss.py` : Label Smoothing
