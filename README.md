# MedicalAI-UAP

This repository contains the codes used in our study on [hogehoge]().

## Terms of use

MIT licensed. Happy if you cite our study when using the codes:

## Usage

```
# Directories
.
├── MedicalAI-UAP
└── dataset
    ├── CellData
    └── ISIC2018_Task3_Training_Input
```

### 1. Check the requirements.

- Python 3.6.6
- tensorflow-gpu 1.12.0
- Keras 2.2.4
- Keras-Applications 1.0.8
- numpy 1.18.2
- pandas 1.0.5
- scikit-learn 0.22.2.post1
- imbalanced-learn 0.6.2
- matplotlib 2.0.2
- Pillow 4.2.1
- tqdm 4.43.0

### 2. Download the following datasets.

- Chest X-ray and OCT images
    - [Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning](https://www.sciencedirect.com/science/article/pii/S0092867418301545?via%3Dihub)
- Diagnosis of melanoma from dermoscopic images
    - [ISIC2018 Task 3: Lesion Diagnosis: Training](https://challenge2018.isic-archive.com/task3/training/)


### 3. Generate the dataset.

```sh
# chestx
python make_data.py --dataset chestx --img_dir ../dataset/CellData/chest_xray

# oct
python make_data.py --dataset oct --img_dir ../dataset/CellData/OCT

# malanoma
python make_data.py --dataset melanoma --img_dir ../dataset/ISIC2018_Task3_Training_Input
```

### 4. Train

```sh
python train_model.py --dataset chestx --model inceptionv3

# `--dataset` argument indicates the dataset: chestx (default), oct or melanoma.
# `--model` argument indicates the model: inceptionv3 (default), vgg16, vgg19, resnet50, inceptionresnetv2, densenet121 or densenet169.
```

### 5. Install the methods for generating universal adversarial perturbations (UAPs).

- `pip install git+https://github.com/hkthirano/adversarial-robustness-toolbox`

### 6. Generate UAPs.

```sh
# non-targeted UAP
python generate_nontargeted_uap.py --dataset chestx

# UAP for targeted attacks to PNEUMONIA
python generate_targeted_uap.py --dataset chestx --target PNEUMONIA
# `--target` argument indicates the target class:
#   when dataset is chestx  , the target class: NORMAL or PNEUMONIA (default).
#   when dataset is oct     , tar target class: CNV, DME, DRUSEN, NORMAL.
#   when dataset is melanoma, tar target class: MEL, NV, BCC, AKIEC, BKL, DF or VASC.

# random UAP
python generate_random_uap.py --dataset chestx
```

### 7. Results

The targeted UAP causes the Inception-v3 models to classify most normal medical images into disease.

- chestx: target PNEUMONIA

    Targeted success rate: [train](result/chestx/conf_mat/train_targeted_inceptionv3_PNEUMONIA_l2_eps0.020.png), [test](result/chestx/conf_mat/test_targeted_inceptionv3_PNEUMONIA_l2_eps0.020.png)

    ![img1](result/chestx/imshow/targeted_inceptionv3_PNEUMONIA_l2_eps0.020.png)

- oct: target CNV

- melanoma: target MEL