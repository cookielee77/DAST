# Domain Adaptive Text Style Transfer

## Introduction
This is a tensorflow implementation of [Domain Adaptive Text Style Transfer](https://arxiv.org/pdf/1908.09395.pdf) by Dianqi Li, Yizhe Zhang, Zhe Gan, Yu Cheng, Chris Brockett, Ming-Ting Sun and Bill Dolan, EMNLP 2019. 

## Environment
The code is based on python 3.6 and tensorflow 1.12 version. The code is developed and tested using one NVIDIA GTX 1080Ti. 

## Data Format
* Data should save in: `/data/${dataset}/train/*.txt`, `/data/${dataset}/valid/*.txt`, `/data/${dataset}/test/*.txt`.
* Taking yelp dataset as an example, please use the below script to generate corresponding format:
```
import codecs
import json

# SAVE DATA
write_train_file = codecs.open('/data/yelp/train.txt', "w", "utf-8")
dict = {"review": line.strip(), "score": score, "other_field_you_want": xxx}
string_ = json.dumps(dict)
write_train_file.write(string_ + '\n')

# LOAD DATA
reader = codecs.open('/data/yelp/train.txt', 'r', 'utf-8')
while True:
    string_ = reader.readline()
    if not string_: break
    dict_example = json.loads(string_)
    review = dict_example["review"]
    score = dict_example["score"]
```
* In each line of `train.txt`, the format will look like: `{"review": "michael is absolutely wonderful .", "score": 1, "something you want"}`.

## Run
In this repo:
`TARGET_DATASET={yelp, amazon}`;
`SOURCE_DATASET={filter_imdb}`;
`NETWORK={CrossAlign, ControlGen}`;
`DA_NETWORK={DAST, DASTC}`.
For more configurations, please see `config.py`.

1. Train binary style classifier:
```
python train_classifier.py --dataset ${TARGET_DATASET}
```

2. Train domain classifier:
```
python train_domain_classifier.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET}
```

3. Train style transfer model on the target domian only:
```
python train_style_transfer.py --dataset ${TARGET_DATASET} --network ${NETWORK}
```

4. Train styel transfer model with domain adaptation:
```
python train_domain_adapt.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET} --network ${DA_NETWORK} --training_portion ${TARGET_DATASET_PORTION}
```
All logs, tensorboard, generated texts will appear in `logs/`

5. Evaluation on generated samples
```
python evaluation.py --domain_adapt --dataset ${TARGET_DATASET} --source_dataset ${SOURCE_DATASET}
```
Note: You need to define the evaluation path `folder_path` by hand in `evaluation.py` file. Sometimes, the pos/neg samples order may be switched, you need to change `line 21-22` in the file. We provide our testing results in `samples` folder for future comparison. 


## Citing
if you find our work is useful in your research, please consider citing: 
```
@InProceedings{li2019domian,
  author = {Li Dianqi and Zhang Yizhe and Gan Zhe and Cheng Yu and Brockett Chris and Sun Ming-Ting and Dolan Bill},
  title     = {Domain Adaptive Text Style Transfer},
  booktitle = {In Proceedings of the Conference on Empirical Methods in Natural Language Processing},
  year      = {2019}
}
```
