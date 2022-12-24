import torch
import shutil
import yaml
import matplotlib as plt

def load_train_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=yaml.Loader)
    return data

def save_checkpoint(state, checkpoint_path, isBest=False, best_model_path=None):
	torch.save(state, checkpoint_path)
	if isBest:
		assert best_model_path==None, 'best_model_path cannot be None'
		shutil.copyfile(checkpoint_path, best_model_path)

def load_checkpoint(best_model_path, checkpoint_path, model, optimizer, isBest=False):
	train_acc_key = 'mot min train acc'
	val_acc_key = 'mot min val acc'

	model, optimizer, epoch, trainacc, valacc, min_valacc = load_model(best_model_path, checkpoint_path, model, optimizer, train_acc_key, val_acc_key, isBest)

	print("optimizer = ", optimizer)
	print("start_epoch = ", epoch)
	print(f'train acc = {trainacc}')
	print(f'val acc = {valacc}')
	print(f'min val acc = {min_valacc}')

	return model, optimizer, min_valacc

def load_model(best_model_path, checkpoint_path, model, optimizer, train_acc_key, val_acc_key, isBest):

	current_ckp = torch.load(checkpoint_path)
	best_ckp = torch.load(best_model_path)
	min_valacc = best_ckp[val_acc_key]

	if isBest:
		model.load_state_dict(best_ckp['state_dict'])
		optimizer.load_state_dict(best_ckp['optimizer'])
		epoch = best_ckp['epoch']
		trainacc = best_ckp[train_acc_key]
		valacc = best_ckp[val_acc_key]

	else:
		model.load_state_dict(current_ckp['state_dict'])
		optimizer.load_state_dict(current_ckp['optimizer'])
		epoch = current_ckp['epoch']
		trainacc = current_ckp[train_acc_key]
		valacc = current_ckp[val_acc_key]	

	return model, optimizer, epoch, trainacc, valacc, min_valacc

def save_checkpoint(state, current_checkpoint_path, is_best=False, best_model_path=None):
    torch.save(state, current_checkpoint_path)
    if is_best:
        assert best_model_path!=None, 'best_model_path should not be None.'
        shutil.copyfile(current_checkpoint_path, best_model_path)

def plot(history, graphType, isTest=False):
    if not isTest:
        plt.plot(history[f'mot_train{graphType}'], label='train', marker= '*')
        plt.plot(history[f'mot_val{graphType}'], label='val', marker = 'o')
    else:
        plt.plot(history[f'mot_test{graphType}'], label='test', marker= '*')
    plt.title(f'{graphType} per epoch')
    plt.ylabel(f'{graphType}')
    plt.xlabel('epoch')
    plt.legend(), plt.grid()
    plt.show()


