import torch.nn as nn
from utils import *
import torch
from time import sleep
from tqdm import tqdm
from torch.autograd import Variable
from torch.optim import *
from tqdm.notebook import tqdm
import wandb
from datetime import datetime

def train(fold, model, device, train_loader, test_loader, num_epochs, totaltrainsamples, totaltestsamples, batch_size, optimizer, lr_scheduler, mot_min_valacc, checkpoint_path, best_model_path, log_df, logpath):
    
    loss = nn.MSELoss()
    acc = nn.L1Loss()
    wandb.watch(model, loss, log='all', log_freq=10)
    
    for epoch in range(num_epochs):
        MTcumuMSE, MVcumuMSE, MTcumuMAE, MVcumuMAE = 0,0,0,0
        model.train()
        
        with tqdm(enumerate(train_loader), total= int(totaltrainsamples/batch_size)) as tepoch:
            for i, (x, y) in tepoch:
                tepoch.set_description(f"Epoch {epoch+1}")
                x = Variable(x).to(device)
                y = Variable(y).to(device)
                # Clear gradients
                optimizer.zero_grad()
                # Forward propagation
                predmot = model(x)
                # Calculate loss and accuracy
                mot_loss = loss(predmot, y)
                mot_acc = acc(predmot, y)
                # Calculating gradients
                mot_loss.backward()
                # Update parameters
                optimizer.step()
                MTcumuMSE += mot_loss.item()
                MTcumuMAE += mot_acc.item()
                
                tepoch.set_postfix(Train_Mot_Loss=mot_loss.item(), Train_Mot_Acc=mot_acc.item())
                sleep(0.1)
                
        lr_scheduler.step()
        
        with torch.no_grad():
            model.eval()
            
            with tqdm(enumerate(test_loader), total= int(totaltestsamples/batch_size)) as valset:
                for i, (x, y) in valset:
                    x = Variable(x).to(device)
                    y = Variable(y).to(device)
                    val_predmot = model(x)
                    mot_loss = loss(val_predmot, y)
                    mot_acc = acc(val_predmot, y)
                    MVcumuMSE += mot_loss.item()
                    MVcumuMAE += mot_acc.item()
                    
                    valset.set_postfix(Val_Mot_Loss=mot_loss.item(), Val_Mot_Acc=mot_acc.item())
                    sleep(0.1)
                    
        MotTrainMSE = MTcumuMSE/len(train_loader)
        MotTrainMAE = MTcumuMAE/len(train_loader)
        MotValMSE = MVcumuMSE/len(test_loader)
        MotValMAE = MVcumuMAE/len(test_loader)
        
        wandb.log(
			{'mot_trainacc': MotTrainMAE,
			'mot_trainloss': MotTrainMSE,
			'mot_valacc': MotValMAE,
			'mot_valloss': MotValMSE})
            
        log_df = log_df.append(
			{'mot_trainacc': MotTrainMAE,
			'mot_trainloss': MotTrainMSE,
			'mot_valacc': MotValMAE,
			'mot_valloss': MotValMSE},
			ignore_index=True)
            
        if (i+1)%1 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]: Train Mot MAE= {MotTrainMAE:.8f}, Train Mot Loss= {MotTrainMSE:.8f}\nVal Mot MAE= {MotValMAE:.8f}, Val Mot Loss= {MotValMSE:.8f}')
            
            checkpoint = {
			'epoch': epoch + 1,
			'mot min train acc': MotTrainMAE,
			'mot min val acc': MotValMAE,
			'state_dict': model.state_dict(),
			'optimizer': optimizer.state_dict(),
			}
            
        save_checkpoint(checkpoint, checkpoint_path)
        
        if MotValMAE <= mot_min_valacc:
            print('Validation Motility acc decreased ({:.8f} --> {:.8f}).  Saving model ...'.format(mot_min_valacc, MotValMAE))
            # save checkpoint as best model
            save_checkpoint(checkpoint, checkpoint_path, True, best_model_path)
            mot_min_valacc = MotValMAE

    now = datetime.now()
    dt_str = now.strftime("%d%m%Y_%H%M")
    log_df.astype('float32').to_csv(f'{logpath}/f{fold}_log_e{epoch}_{dt_str}.csv')
    print('Finished training')
    
    return log_df
    
    

