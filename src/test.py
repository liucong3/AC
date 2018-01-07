# import torch, torch.nn.functional as F
# from torch.autograd import Variable as V

# a = torch.randn(3,4)
# # print (a == a.max()).type_as(a)
# b = F.softmax(V(a)).data
# # print b.sum(1)
# # print b.sum(1)
# # print b.sum(2)

# # import numpy
# # policy = [0,0.1,0.1,0.1,0.7]
# # for i in range(30):
# # 	index = numpy.random.choice(range(len(policy)), size=1, p=policy)[0]
# # 	print index

# import torch

# i = (torch.rand(6,1,4,5) * 3).long()
# print i

# a = torch.zeros(6,3,4,5)
# # k = i.view(6,1,-1)
# # a.view(6,3,-1).scatter_(1, k, torch.ones(k.size()))
# a.scatter_(1, i, torch.ones(i.size()))
# print a


import threading
from time import sleep

def fun():
	while True:
		print('+', threading.currentThread().name)
		sleep(1.5)

threads = []
for i in range(5):
	threads.append(threading.Thread(target=fun))
	threads[-1].setDaemon(True)
for i in range(5):
	threads[i].start()
sleep(5)
