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
```bash
from google.colab import drive
drive.mount('/content/gdrive')
```
2. Copy the folder containing dataset to Colab
```bash
%cp -av 'your_drive_folder_here' 'your_destination_here'
```
3. Install wandb (create an account first)
```bash
!pip install wandb -qqq
```
### Train
```bash
%run main.py --fold 1
Compulsory argument:
    [--fold]    :   number of fold for training and validation
                    type int

Optional arguments:
    [--epoch]   :   Number of epochs for training, default=50
                    type int
    [--isContinue]: If you wish to train from previous stopped checkpoint
```
Example: 
For fold 1, continue training from previous checkpoint, which stopped at epoch 50, continue for another 20 epochs.
```bash
%run main.py --fold 1 --isContinue --epoch 20
```
### Test
```bash
%run test.py --fold 1 --isTestSet --testPath 'your_path_here'
Compulsory argument:
    [--fold]    :   number of fold for training and validation
                    type int
    [--isTestSet] or [--isSingleTest]:  type of test samples. If test sample is more than one, please prepare a csv file with headings same as the directory csv file used for training/validation, examples in the folder "dir"

Optional arguments:
    [--epoch]   :   Number of epochs for training, default=50
                    type int
    [--isContinue]: If you wish to train from previous stopped checkpoint
```
```