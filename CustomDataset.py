import numpy as np
import torch as torch
from torch.utils.data import Dataset

class CustomImageDataset(Dataset):
	def __init__(self, dataframe):
		self.dataframe = dataframe
		self.list_ID = self.dataframe[['ORI_DIR1','ORI_DIR2']]
		self.yvalue = self.dataframe[['PR', 'NPR', 'IM', 'CONC']]

	def __len__(self):
		return len(self.list_ID)

	def __getitem__(self, index):
		data_path = self.list_ID.iloc[index,0]
		X1 = (np.load(data_path))
		y1 = np.array(self.yvalue.iloc[index,0:3])
		return torch.from_numpy(X1).float(), torch.from_numpy(y1).float()