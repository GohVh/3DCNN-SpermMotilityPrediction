from utils import *
import torch
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

def predict(model, test_target):
    with torch.no_grad():
        model.eval()
        predmot = model(test_target)
        predmot = (predmot.cpu().numpy())*100
        result={f'Progressive: {predmot[0]}%, Non-progressive: {predmot[1]}%, Immotile: {predmot[2]}%'}
    
    return result

def test(model, device, test_loader, log_df):
    result = pd.DataFrame(columns=['PR', 'NPR', 'IM'])
    loss = nn.MSELoss()
    acc = nn.L1Loss()
    MTcumuMSE, MTcumuMAE = 0, 0

    with torch.no_grad():
        model.eval()
        with tqdm(enumerate(test_loader)) as testset:
                for i, (x, y) in testset:
                    x = Variable(x).to(device)
                    y = Variable(y).to(device)
                    test_predmot = model(x)
                    mot_loss = loss(test_predmot, y)
                    mot_acc = acc(test_predmot, y)
                    MTcumuMSE += mot_loss.item()
                    MTcumuMAE += mot_acc.item()

                    test_predmot_actual = (test_predmot.cpu().numpy())*100

                    log_df = log_df.append(
                        {'mot_testacc':MotTestMAE,
                        'mot_testloss': MotTestMSE},
                        ignore_index=True)
                        
                    result = result.append({'PR':test_predmot_actual[0], 'NPR':test_predmot_actual[1], 'IM':test_predmot_actual[2]}, ignore_index=True)
                    
                    testset.set_postfix(Test_Mot_Loss=mot_loss.item(), Test_Mot_Acc=mot_acc.item())
                    sleep(0.1)
                    
        MotTestMSE = MTcumuMSE/len(test_loader)
        MotTestMAE = MTcumuMAE/len(test_loader)
        
        print(f'Test Mot MAE= {MotTestMAE:.8f}, Test Mot Loss= {MotTestMSE:.8f}')            
        log_df.astype('float32').to_csv(f'{PATH["MODEL_DIR"]}/f{args.fold}_test_log.csv')

    return result

parser = argparse.ArgumentParser()
config = load_train_config('./config.yaml')
parser.add_argument('--fold', required=True, type=int, help='Fold number for training and testing. Keyword: 1, 2, 3')
# parser.add_argument('--isContinue', action='store_true', default=False, help='boolean, True False')
parser.add_argument('--isSingleTest', action='store_true', default=False, help='boolean, True False')
parser.add_argument('--singlePath', default=None, required=True, help='the directory of the test file')
parser.add_argument('--isTestSet', action='store_true', default=False, help='boolean, True False')
parser.add_argument('--testPath', default=None, required=True, help='the csv file containing the directories of files to be tested, in the format as specified in readme.')
# parser.add_argument('--epoch', default=50, type=int, required = True, help='No. of epoch')
args = parser.parse_args()

# wandb.init(project='3DCNN-Sperm Motility Prediction', config=config)
globals().update(config)
# wandb.config.update(config)
 
def main():
    
    # set device to cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # init model & param
    model = CNN3DModel().to(device)
    min_valacc = np.Inf
    optimizer = torch.optim.Adam(model.parameters(), lr=PARAM["INITIAL_LR"])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=PARAM["STEP_SIZE"], gamma=PARAM["GAMMA"], last_epoch=-1)
    
    # define directory
    checkpoint_path = f'{PATH["MODEL_DIR"]}_f{args.fold}_checkpoint.pth'
    best_model_path = f'{PATH["MODEL_DIR"]}_f{args.fold}_bestmodel.pth'

    # load model with best validation accuracy
    model, optimizer, min_valacc = load_checkpoint(best_model_path, checkpoint_path, model, optimizer, isBest=True)

    result=None
    
    if args.isSingleTest:
        assert args.singlePath==None, 'directory of the tested file cannot be None, please specified as "--singlePath your_path_here"'
        testfile = np.load(args.singlePath)
        result = predict(model, testfile)
        print(result)

    if args.isTestSet:
        assert args.testPath==None, "the csv file's path containing directories of tested files cannot be None, prepare the csv directory as mentioned in readme, then specified as '--testPath your_path_here'"
        testfile = f'{args.testPath}'

        # data generation
        testdf = pd.read_csv(testfile)
        test_data = CustomImageDataset(testdf)
        test_loader = DataLoader(test_data, shuffle=False)
        
        # generate log
        test_log_df = pd.DataFrame({'mot_testacc': [], 'mot_testloss': []})
        
        result = test(model, device, test_loader, test_log_df)
        result.astype('float32').to_csv(f'{PATH["MODEL_DIR"]}/f{args.fold}_test_result.csv')
        plot(test_log_df, 'acc', isTest=True)
        plot(test_log_df, 'loss', isTest=True)

if __name__ == '__main__':
	main()
    
# def predict(model, device, test_target):
