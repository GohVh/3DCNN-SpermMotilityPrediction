<!-- readme -->
# 3D Convolutional Neural Networks for Sperm Motility Prediction
**This is the official implementation of 3D Convolutional Neural Networks for Sperm Motility Prediction, using stacked dense optical flow frames as input.**

## Prepare & Preprocess Dataset
Refer to the prepare_dataset notebook

## Using a notebook server
### Conda Environment
Create conda environment with yml file.
```bash
conda env create -f env.yml
```

## Using Google Colab
### Load dataset
Copy the dataset saved in Google Drive to Colab (Samples loading during training is slower if load directly from Drive)
1. Mount your drive
2. Copy the folder containing dataset to Colab
3. Install wandb (create an account first)
```bash
from google.colab import drive
drive.mount('/content/gdrive')
```
```bash
%cp -av 'your_drive_folder_here' 'your_destination_here'
```
```bash
!pip install wandb -qqq
```
### Train
```bash
!pip install wandb -qqq
```