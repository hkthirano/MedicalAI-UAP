# MedicalAI-UAP

# ディレクトリの配置

```
.
├── MedicalAI-UAP
└── dataset
    ├── CellData
    ├── ISIC2018_Task3_Training_GroundTruth
    └── ISIC2018_Task3_Training_Input
```

# データ

- chestx・oct : [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)
- melanoma : [ISIC2018 Task 3: Lesion Diagnosis: Training](https://challenge2018.isic-archive.com/task3/training/)

## 前処理

- chestx : `python make_chestx_dataset.py`

- oct : `python make_oct_dataset.py`

- melanoma : `python make_melanoma_dataset.py`

# モデル学習

```sh
python train_model.py --dataset chestx --model inceptionv3 --gpu 0

# dataset : chestx, oct, melanoma
# model : inceptionv3, vgg16, vgg19, resnet50, inceptionresnetv2, densenet121, densenet169
```

# ノイズ作成