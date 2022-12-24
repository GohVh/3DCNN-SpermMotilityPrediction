# importing the libraries
from statistics import mean
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import *
from test import *

import wandb
import argparse
wandb.login()

from utils import *
from CustomDataset import CustomImageDataset
from model import *
from selftorchsummary import summary

parser = argparse.ArgumentParser()
config = load_train_config('./config.yaml')
parser.add_argument('--fold', required=True, type=int, help='Fold number for training and testing. Keyword: 1, 2, 3')
parser.add_argument('--isContinue', action='store_true', default=False, help='boolean, True False')
parser.add_argument('--epoch', default=50, type=int, required = True, help='No. of epoch')
args = parser.parse_args()

wandb.init(project='3DCNN-Sperm Motility Prediction', config=config)
globals().update(config)
wandb.config.update(config)

def main():

	# set device to cuda
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	# define directory
	checkpoint_path = f'{PATH["MODEL_DIR"]}/f{args.fold}_checkpoint.pth'
	best_model_path = f'{PATH["MODEL_DIR"]}/f{args.fold}_bestmodel.pth'
	trainfile = f'{PATH["MODEL_DIR"]}/trainf{args.fold}.csv'
	testfile = f'{PATH["MODEL_DIR"]}/testf{args.fold}.csv'

	# data generation
	traindf, testdf = pd.read_csv(trainfile), pd.read_csv(testfile)
	# Normalized and scale data
	col_mot = ['PR','NPR','IM']
	traindf[col_mot] = traindf[col_mot].apply(lambda x: x/100)
	testdf[col_mot] = testdf[col_mot].apply(lambda x: x/100)

	# init model & param
	model = CNN3DModel().to(device)
	min_valacc = np.Inf
	optimizer = torch.optim.Adam(model.parameters(), lr=PARAM["INITIAL_LR"])
	lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=PARAM["STEP_SIZE"], gamma=PARAM["GAMMA"], last_epoch=-1)

	train_data = CustomImageDataset(traindf)
	test_data = CustomImageDataset(testdf)
	train_loader = DataLoader(train_data, batch_size=PARAM["BATCH_SIZE"], shuffle=False)
	test_loader = DataLoader(test_data, batch_size=PARAM["BATCH_SIZE"], shuffle=False)
	totaltrainsamples = traindf.shape[0]
	totaltestsamples = testdf.shape[0]

	# generate log
	log_df = pd.DataFrame({'mot_trainacc': [], 'mot_valacc': [], 'mot_trainloss': [], 'mot_valloss': []})

	# check if its start from previous trained checkpoint
	if args.isContinue:
		model, optimizer, min_valacc = load_checkpoint(best_model_path, checkpoint_path, model, optimizer, isBest=False)
	
	train(args.fold, model, device, train_loader, test_loader, args.epoch, totaltrainsamples, totaltestsamples, PARAM["BATCH_SIZE"], optimizer, lr_scheduler, min_valacc, checkpoint_path, best_model_path, log_df)

	plot(log_df, 'acc', isTest=False)
	plot(log_df, 'loss', isTest=False)

if __name__ == '__main__':
	main()



# def generate_param(training_status, optimizer_type, change_lr_rate, model, model_type, checkpoint_type, initial_lr, best_model_path, checkpoint_path):
	
# 	tempmodel = model
# 	min_valacc = np.Inf

# 	if optimizer_type == 'sgd':
# 		optimizer = torch.optim.SGD(tempmodel.parameters(), lr=initial_lr)
# 		print(optimizer)
# 	elif optimizer_type == 'adam':
# 		optimizer = torch.optim.Adam(tempmodel.parameters(), lr=initial_lr)
# 		print(optimizer)

# 	if training_status == 'continue':

# 		tempmodel, optimizer, min_valacc = load_checkpoint(best_model_path, checkpoint_path, tempmodel, optimizer, model_type=model_type, checkpoint_type=checkpoint_type)

# 		if change_lr_rate:
# 			optimizer.param_groups[0]['initial_lr'] = initial_lr
# 			optimizer.param_groups[0]['lr'] = initial_lr
# 			print(f'updated optimizer:\n{optimizer}')

# 	return tempmodel, optimizer, min_valacc

# define argument
	# parser = argparse.ArgumentParser(description='PyTorch Motility and Concentration Prediction')
	# parser.add_argument('--batch_size', type=int, default=30, help='batch size (default: 10)')
	# parser.add_argument('--step_size', type=int, default=10, help='step size for lr scheduler, default = 10')
	# parser.add_argument('--gamma', type=float, default=0.95, help='gamma for adam optimizer, default = 0.95')
	# parser.add_argument('--weight_decay', type=float, default=1e-5)

	# arguments required user define
	# parser.add_argument('--fold', type=int, required = True, help='Fold number for training and testing. Keyword: 1, 2, 3')
	# parser.add_argument('--initial_lr', type=float, required = True, help='initial learning rate for Adam optimizer')
	# parser.add_argument('--training_status', type=str, required = True, help='Keyword: init, continue')
	# parser.add_argument('--isBest', action='store_true', default=False)
	# parser.add_argument('--isContinue', action='store_true', default=False)
	# parser.add_argument('--change_lr_rate', action='store_true', default=False)
	# parser.add_argument('--checkpoint_type', type=str, required = True, help='Load statedict from best model or current model. Keyword: best, current')
	# parser.add_argument('--epoch', type=int, required = True, help='No. of epoch')
	# parser.add_argument('--train_test', type=str, required=True, help='Train or test required? Keyword: train, test')
	# parser.add_argument('--project_name', type=str, required=True, default='3dcnn', help='wandb project name')
	# parser.add_argument('--lrscheduler_type', type=str, required=True, help='lr scheduler type')
	# parser.add_argument('--optimizer_type', type=str, required=True, help='optimizer type')
	
	# args = parser.parse_args()

# checkpoint_path = f'/content/gdrive/MyDrive/share/{args.model_type}_f{args.fold}/checkpoint.pth'
	# best_model_path = f'/content/gdrive/MyDrive/share/{args.model_type}_f{args.fold}/bestmodel.pth'
	# trainfile = f'/content/trainf{args.fold}.csv'
	# testfile = f'/content/testf{args.fold}.csv'