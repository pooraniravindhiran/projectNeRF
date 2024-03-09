import torch.nn as nn

class tinyNerf(nn.Module):
	'''
      # Input is of shape - (h * w * num_samples, (2 * num_posencoding_functions * 3)+(2 * num_direncoding_functions * 3))
      # Output is of shape - (h * w * num_samples, 4) where the 4-dim vector represents the RGB information and density of that respective 3D sample point.
  	'''
	def __init__(self, num_encoding_functions: int, filter_size=128):
		super(tinyNerf, self).__init__()
		self.fc1 = nn.Linear(3 +(2 * num_encoding_functions * 3), filter_size)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(filter_size, filter_size)
		self.relu2 = nn.ReLU()
		self.fc3 = nn.Linear(filter_size, 4)
	
	def forward(self, x):
		x = self.fc1(x)
		x = self.relu1(x)
		x = self.fc2(x)
		x = self.relu2(x)
		x = self.fc3(x)
		return x