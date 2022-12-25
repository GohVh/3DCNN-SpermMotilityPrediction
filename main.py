# importing the libraries
from statistics import mean
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from train import *

import wandb
import argparse
wandb.login()

from utils import *
from CustomDataset import CustomImageDataset
from model import *

parser = argparse.ArgumentParser()
config = load_train_config('./config.yml')
parser.add_argument('--fold', required=True, type=int, help='Fold number for training and testing. Keyword: 1, 2, 3')
parser.add_argument('--isContinue', action='store_true', default=False, help='boolean, True False')
parser.add_argument('--epoch', default=10, type=int, help='No. of epoch')
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
	print(f'log_df: {log_df}')

	# check if its start from previous trained checkpoint
	if args.isContinue:
		model, optimizer, min_valacc = load_checkpoint(best_model_path, checkpoint_path, model, optimizer, isBest=False)
	
	log_df = train(args.fold, model, device, train_loader, test_loader, args.epoch, totaltrainsamples, totaltestsamples, PARAM["BATCH_SIZE"], optimizer, lr_scheduler, min_valacc, checkpoint_path, best_model_path, log_df, PATH["MODEL_DIR"])

	plot(log_df, 'acc', isTest=False)
	plot(log_df, 'loss', isTest=False)

if __name__ == '__main__':
	main()