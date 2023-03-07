<!-- readme -->
# 3D Convolutional Neural Networks for Sperm Motility Prediction
**This is the official implementation of 3D Convolutional Neural Networks for Sperm Motility Prediction, using stacked dense optical flow frames as input.**

This work is presented in 2nd International Conference on Intelligent Cybernetics Technology & Applications 2022 (ICICyTA) and published in IEEE Xplore. [Link](https://ieeexplore.ieee.org/abstract/document/10037950)

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
```
```console
Compulsory argument:
    [--fold]
    number of fold for training and validation
    type int

Optional arguments:
    [--epoch]
    Number of epochs for training, default=50
    type int

    [--isContinue]
    If you wish to train from previous stopped checkpoint
```
Example: 

For fold 1, continue training from previous checkpoint, which stopped at epoch 50, continue for another 20 epochs.
```bash
%run main.py --fold 1 --isContinue --epoch 20
```
### Test
```bash
%run test.py --fold 1 --isTestSet --testPath 'your_path_here'
```
```console
Compulsory argument:
    [--fold]
    number of fold for training and validation
    type int

    [--isTestSet] or [--isSingleTest]
    Type of test samples. 
    If test sample is more than one, use --isTestSet.
    Please prepare a csv file with headings same as the directory csv file used for training/validation.
    Examples in the folder "dir".
    If test sample is only one, use --isSingleTest

    [--testPath] or [--singlePath]
    File type for --testPath = .csv
    File type for --singlePath = .npy
```
Example:

For fold 1, a test set is prepared with directories as specified.
```bash
%run test.py --fold 1 --isTestSet --testPath 'your_path_here.csv'
```
For fold 1, a test sample is prepared in the format of .npy
```bash
%run test.py --fold 1 --isSingleTest --singlePath 'your_path_here.npy'
```
## Result
The results shown below are predicted using model trained from Dataset Type C, Fold 1.
<div align="center"> 
  <img src="https://github.com/GohVh/3DCNN-SpermMotilityPrediction/blob/main/result.JPG" alt="screenshot" />
</div>

### :handshake: Contact
- LinkedIn Profile: [GohVh](https://www.linkedin.com/in/gohvh95/)
- Email: gohvh95@gmail.com

### :gem: Acknowledgements

The dataset used in this project is an open-source Multimodal Video Dataset of Human Spermatozoa. It is a multi-modal dataset containing different data sources such as videos, biological analysis data, and participant data. It consists of anonymized data from 85 different participants, of which the samples collection and analysis approach were performed according to WHO recommendation. This dataset is prepared by a team of members from Simula Research Laboratory and SimulaMet.

[Simula Dataset](https://datasets.simula.no/visem/)

[Simula Official Github Repo](https://github.com/simula/datasets.simula.no)
