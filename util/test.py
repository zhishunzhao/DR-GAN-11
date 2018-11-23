import torch.nn.functional as F
import torch
import math
from torch.autograd import Variable
torch.manual_seed(7)

loss_fn = torch.nn.BCELoss(reduce=False, size_average=False)
input = Variable(torch.randn(3, 4))
target = Variable(torch.FloatTensor(3, 4).random_(2))
loss = loss_fn(F.sigmoid(input), target)
print('input is: ', input)
print('target is: ', target)
print('loss is: ', loss)
a = torch.FloatTensor((-0.1468))
print('Number one: ', math.log(F.sigmoid(a)) * 1)