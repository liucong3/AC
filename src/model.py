import torch, torch.nn as nn, torch.nn.functional as F
from torch.autograd import Variable

# 3x3 Convolution
def conv3x3(in_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False):
	return nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
					 stride=stride, padding=padding, bias=bias)

# conv + batch_nor + [ relu ]
def convBlock(in_channels, out_channels, non_linearity, kernel_size=3, padding=1):
	elems = [ conv3x3(in_channels, out_channels, kernel_size=kernel_size, padding=padding), 
			nn.BatchNorm2d(out_channels) ]
	if non_linearity is not None:
		elems.append( non_linearity )
	return nn.Sequential(*elems)


# residual block
class ResidualBlock(nn.Module):

	def __init__(self, channels):
		super(ResidualBlock, self).__init__()
		self.relu = nn.ReLU(inplace=True)
		self.conv1 = convBlock(channels, channels, self.relu)
		self.conv2 = convBlock(channels, channels, None)

	def forward(self, x):
		residual = x
		x = self.conv1(x)
		x = self.conv2(x)
		x += residual
		x = self.relu(x)
		return x

class View(nn.Module):

	def __init__(self, *size):
		super(View, self).__init__()
		self.size = size

	def forward(self, x):
		return x.view(*self.size)

class AlphaGoModel(nn.Module):

	def __init__(self, in_channels, out_channels, hidden_channels, residual_blocks, board_size=(9,10)):
		super(AlphaGoModel, self).__init__()
		# input conv
		self.relu = nn.ReLU(inplace=True)
		self.conv_in = convBlock(in_channels, hidden_channels, self.relu)
		# residual bocks
		elems = []
		for i in range(residual_blocks):
			elems.append(ResidualBlock(hidden_channels))
		self.residual_blocks = nn.Sequential(*elems)
		# pilicy head
		self.policy_head = nn.Sequential( convBlock(hidden_channels, hidden_channels, self.relu),
				conv3x3(hidden_channels, out_channels, bias=True) )
		# value head
		self.policy_size = (out_channels, board_size[0], board_size[1])
		self.value_head = nn.Sequential( convBlock(hidden_channels, 1, self.relu, kernel_size=1, padding=0),
				View(-1, board_size[0] * board_size[1]),
				nn.Linear(board_size[0] * board_size[1], 64),
				self.relu,
				nn.Linear(64, 1),
				nn.Tanh() )

	def forward(self, x):
		x = self.conv_in(x)
		x = self.residual_blocks(x)
		p = F.softmax(self.policy_head(x))
		v = self.value_head(x)
		return p, v

	def loss(self, p, v, target_p, target_v):
		batch_size = p.size(0)
		p = p.view(batch_size, -1).log()
		target_p = target_p.view(batch_size, -1)
		return (((v - target_v) ** 2).sum() - (p * target_p).sum()) / batch_size

