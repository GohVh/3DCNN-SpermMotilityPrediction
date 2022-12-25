import torch.nn as nn

class CNN3DModel(nn.Module):
	def __init__(self):
		super().__init__()

		self.part1 = nn.Sequential(
			self._conv_blocks(3, 64),
			self._conv_blocks(64, 64),
			self._conv_blocks(64, 128),
			nn.AvgPool3d(kernel_size=(7, 7, 1)),
			nn.Flatten()
			)

		self.part2 = nn.Sequential(
			nn.ReLU(),
			nn.Linear(512,3),
			nn.Softmax()
			)

	def _conv_blocks(self, inputc, outc):
		conv_layers = nn.Sequential(
			nn.Conv3d(inputc, outc, kernel_size=(3, 3, 3), padding='same'),
			nn.ReLU(),
			nn.MaxPool3d((2, 2, 2)),
			nn.BatchNorm3d(outc))
		return conv_layers

	def forward(self, x):
		out = self.part2(self.part1(x))
		return out

