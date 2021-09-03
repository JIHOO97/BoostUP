# Boostcamp Image Classification Competition
Competition의 목적은 사람 얼굴 이미지 데이터만을 가지고 마스크를 썼는지, 쓰지 않았는지, 정확히 쓴 것인지를 분류하는것에 있었습니다. 하지만 추가적으로 나이대, 성별 특성을 추가하여 총 18개의 분류로 나누는 것이 최종적인 목표가 되었습니다.

대회 사이트 [AI stage](https://stages.ai/)

## Hardware
AI stage에서 제공한 server, GPU
- GPU: V100

## 개요
1. [Installation](#installation)
2. [Download Official Image](#download-official-image)
3. [Make RGBY Images](#make-rgby-images) for official.
4. [Download Pretrained models](#pretrained-models)
5. [Inference](#inference)
6. [Make Submission](#make-submission)

## Installation
다음과 같은 명령어로 필요한 libraries를 다운 받습니다.
```
pip install -r requirements.txt
```

## Dataset
데이터셋은 2700명의 사람들이 각각 마스크를 안 쓴 사진 1장, 쓴 사진 5장, 제대로 쓰지 않은 사진 1장으로 되어있습니다.
데이터는 공개 할 수 없습니다.

## Data preprocessing (label.ipynb)
label 파일은 데이터의 노이즈를 제거해주는 파일입니다.
  1. 이미지를 한장한장 띄워주고, 해당 이미지의 설명과 다른 이미지가 있다면 (1)을 눌러 wrong images폴더에 넣어줍니다.
  2. wrong images폴더 안에있는 이미지들을 띄워주어 다시 한번 검토하여 불필요한 이미지가 있다면 삭제합니다.
  3. wrong images폴더에 남은 이미지들을 다시 띄워주어 해당 이미지가 어느 class에 해당되는지 숫자를 치면, 자동으로 해당 class로 재분류가 됩니다.
  4. 재분류한 이미지들을 csv파일로 만들어 줍니다. (이미지의 절대 경로와 class를 저장합니다)


## Create separate models for age, mask, and gender (three_model.ipynb)
model: resnet152

설명
  1. resnet152 pretrained 모델에서 마지막 fully connected layer를 제거
  2. 3개의 다른 fully connected layers를 생성하여 각각 age, mask, gender에 적용
  3. 3개의 다른 fully connected layers에서 나온 결과값을 합하여 final output을 생성

## Create a single model for all age, mask, and gender (one_model.ipynb)
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
resnet152 | 
vit_base_patch16_224 | 
custom_model | 

### Search augmentation
To find suitable augmentation, 256x256 image and resnet18 are used.
It takes about 2 days on TitanX. The result(best_policy.data) will be located in *results/search* directory.
The policy that I used is located in *data* directory.
```
$ python train.py --config=configs/search.yml
```

### Train models
To train models, run following commands.
```
$ python train.py --config={config_path}
```
To train all models, run `sh train.sh`

The expected training times are:

Model | GPUs | Image size | Training Epochs | Training Time
------------ | ------------- | ------------- | ------------- | -------------
resnet34 | 1x TitanX | 512 | 40 | 16 hours
inception-v3 | 3x TitanX | 1024 | 27 | 1day 15 hours
se-resnext50 | 2x TitanX | 1024 | 22 | 2days 15 hours

### Average weights
To average weights, run following commands.
```
$ python swa.py --config={config_path}
```
To average weights of all models, simply run `sh swa.sh`
The averages weights will be located in *results/{train_dir}/checkpoint*.

### Pretrained models
You can download pretrained model that used for my submission from [link](https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz?dl=0). Or run following command.
```
$ wget https://www.dropbox.com/s/qo65gw8kml5hgag/results.tar.gz
$ tar xzvf results.tar.gz
```
Unzip them into results then you can see following structure:
```
results
  +- resnet34.0.policy
  |  +- checkpoint
  +- resnet34.1.policy
  |  +- checkpoint
  +- resnet34.2.policy
  |  +- checkpoint
  +- resnet34.3.policy
  |  +- checkpoint
  +- resnet34.4.policy
  |  +- checkpoint
  +- inceptionv3.attention.policy.per_image_norm.1024
  |  +- checkpoint
  +- se_resnext50.attention.policy.per_image_norm.1024
  |  +- checkpoint
```

## Inference
If trained weights are prepared, you can create files that contains class probabilities of images.
```
$ python inference.py \
  --config={config_filepath} \
  --num_tta={number_of_tta_images, 4 or 8} \
  --output={output_filepath} \
  --split={test or test_val}
```
To make submission, you must inference test and test_val splits. For example:
```
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test_val.csv --split=test_val
$ python inference.py --config=configs/resnet34.0.policy.yml --num_tta=8 --output=inferences/resnet34.0.test.csv --split=test
```
To inference all models, simply run `sh inference.sh`

## Make Submission
Following command will ensemble of all models and make submissions.
```
$ python make_submission.py
```
If you don't want to use, modify *make_submission.py*.
For example, if you want to use inception-v3 and se-resnext50 then modify *test_val_filenames, test_filenames and weights* in *make_submission.py*.
```
test_val_filenames = ['inferences/inceptionv3.0.test_val.csv',
                      'inferences/se_resnext50.0.test_val.csv']
                      
test_filenames = ['inferences/inceptionv3.0.test.csv',
                  'inferences/se_resnext50.0.test.csv']
                  
weights = [1.0, 1.0]
```
The command generate two files. One for original submission and the other is modified using data leak.
- submissions/submission.csv
- submissions/submission.csv.leak.csv
