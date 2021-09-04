# Boostcamp Image Classification Competition
Competition의 목적은 사람 얼굴 이미지 데이터만을 가지고 마스크를 썼는지, 쓰지 않았는지, 정확히 쓴 것인지를 분류하는것에 있었습니다. 하지만 추가적으로 나이대, 성별 특성을 추가하여 총 18개의 분류로 나누는 것이 최종적인 목표가 되었습니다.

대회 사이트 [AI stage](https://stages.ai/)

## Hardware
AI stage에서 제공한 server, GPU
- GPU: V100

## 개요
1. [Installation](#installation)
2. [Data preprocessing](#data-preprocessing)
3. [3 models](#create-separate-models-for-age-mask-and-gender)
4. [1 model](#create-a-single-model-for-all-age-mask-and-gender)
5. [Voting](#voting)
6. [Weight Check and Bacward Graph plot](#weight-check-and-bacward-graph-plot)
7. [References](#references)
## Installation
다음과 같은 명령어로 필요한 libraries를 다운 받습니다.
```
pip install -r requirements.txt
```

## Dataset
데이터셋은 2700명의 사람들이 각각 마스크를 안 쓴 사진 1장, 쓴 사진 5장, 제대로 쓰지 않은 사진 1장으로 되어있습니다.
데이터는 공개 할 수 없습니다.

## Data preprocessing
파일: label.ipynb

label 파일은 데이터의 노이즈를 제거해주는 파일입니다.
  1. 이미지를 한장한장 띄워주고, 해당 이미지의 설명과 다른 이미지가 있다면 (1)을 눌러 wrong images폴더에 넣어줍니다.
  2. wrong images폴더 안에있는 이미지들을 띄워주어 다시 한번 검토하여 불필요한 이미지가 있다면 삭제합니다.
  3. wrong images폴더에 남은 이미지들을 다시 띄워주어 해당 이미지가 어느 class에 해당되는지 숫자를 치면, 자동으로 해당 class로 재분류가 됩니다.
  4. 재분류한 이미지들을 csv파일로 만들어 줍니다. (이미지의 절대 경로와 class를 저장합니다)


## Create separate models for age, mask, and gender
파일: 3models_combined.ipynb

model used: resnet152
  1. resnet152 pretrained 모델에서 마지막 fully connected layer를 제거
  2. 3개의 다른 fully connected layers를 resnet152 마지막 layer에 더해주어, 각각 age, mask, gender에 적용
  3. 3개의 다른 fully connected layers에서 나온 결과값을 합하여 final output을 생성

## Create a single model for all age, mask, and gender
파일: one_model.ipynb

다음과 같이 wandb를 설정해주세요.
```
wandb.init(project='your-project-name', entity='your-entity-name',config = {
    'learning_rate':0.001,
    'batch_size':16,
    'epoch':2,
    'model':'your-model-name',
    'momentum':0.9,
    'img_x':img_size_x[2],
    'img_y':img_size_y[2],
    'kfold_num':3,
})
config = wandb.config
```

Model | GPUs | Image size | Training Epochs | k-fold | batch size | learning_rate | momentum
------------ | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | -------------
resnet152 | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9
vit_base_patch16_224 | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9
custom_model | V100 | 224x224 | 2 | 3 | 16 | 0.001 | 0.9

Model | Test Accuracy
------------ | -------------
vit_base_patch16_224 | 92.12
resnet152 | 91.64
custom_model | 2.51

Test dataset을 만들어서 위에서 만든 모델로 eval images에 대한 답을 추출한다.

Model | Eval Accuracy (test) | Eval F1 score (test) | Eval Accuracy (final) | Eval F1 score (final)
------------ | ------------- | ------------- | ------------- | -------------
resnet152 | 80.460 | 0.774 | 79.937 | 0.755
vit_base_patch16_224 | 79.952 | 0.766 | 79.619 | 0.756

## Voting
파일: voting.ipynb

가장 성능이 좋았던 10개의 모델을 불러내어 softvoting하여 output추출

Combined Model (resnet의 결과값에 가중치 1, vit의 결과값에 가중치 0.625)
- resnet152
- resnet50
- resnet50 (complex transformation applied)
- resnet34
- resnet34 (mean and std for each image)
- vit_base_patch16_224(kfold5,epoch2, batch64)
- vit_base_patch16_224(stratified-kfold5, epoch1, cutmix-beta1, batch64)
- vit_base_patch16_224(kfold5, epoch1, batch64) 
- vit_base_patch16_224(stratified-kfold3, epoch5,  cutmix-beta1,batch64, swa)
- vit_large_patch16_224(stratified-kfold5, epoch1, batch 16)

Eval Accuracy (test) | Eval F1 score (test) | Eval Accuracy (final) | Eval F1 score (final)
------------ | ------------- | ------------- | -------------
81.635 | 0.781 | 81.000 | 0.771

## Weight Check and Bacward Graph plot
https://kmhana.tistory.com/25

파일:clasify_module.ipynb

모델에 대한 gradient 값을 출력하고 , backward graph plot

## References
https://arxiv.org/pdf/1812.01187.pdf
https://kmhana.tistory.com/25
