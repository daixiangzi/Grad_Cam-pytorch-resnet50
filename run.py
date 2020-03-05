import os
import shutil
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.models as models
from config import Config
from MyDataset import MyDataset
from utils import  AverageMeter,accuracy

opt = Config()
#0:cat,1:dog, default set on config.py
if not os.path.isdir(opt.save_dir):
        os.makedirs(opt.save_dir)
use_cuda = torch.cuda.is_available()
# Random seed
if opt.seed is None:
    opt.seed = random.randint(1, 10000)
random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
if use_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(opt.seed)

best_acc = 0  # best test accuracy
def train_test(dataloader, model, criterion,use_cuda,train_flag=True):
	losses = AverageMeter()
	top1 = AverageMeter()
	if train_flag:
		model.train()
	else:
		model.eval()
	for batch_idx, (inputs, targets) in enumerate(dataloader):
		if use_cuda:
			inputs, targets = inputs.cuda(), targets.cuda(async=True)
		inputs, targets = torch.autograd.Variable(inputs), torch.autograd.Variable(targets)
		outputs = model(inputs)
		loss = criterion(outputs, targets)
		prec1,_= accuracy(outputs.data, targets.data,topk=(1,2))
		losses.update(loss.item(), inputs.size(0))
		top1.update(prec1.item(), inputs.size(0))
		if train_flag:
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
	return (losses.avg, top1.avg)
	

# test and train transform 
transform_train = transforms.Compose([
            
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

transform_test = transforms.Compose([
	transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
# load data
trainset = MyDataset(opt.train_data,transform=transform_train)
trainloader = data.DataLoader(trainset, batch_size=opt.train_batch, shuffle=True, num_workers=opt.workers)

testset = MyDataset(opt.test_data,transform=transform_test)
testloader = data.DataLoader(testset, batch_size=opt.test_batch, shuffle=False, num_workers=opt.workers)

#define model
#Support other model on https://github.com/pytorch/vision/tree/master/torchvision/models
model = models.resnet50(pretrained=False,num_classes=opt.num_class)
#default single GPU model
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
model = model.to(device)
#if use_cuda:
#	model = torch.nn.DataParallel(model).cuda()

#define optimizer
if opt.optim=="SGD":
	optimizer = optim.SGD(model.parameters(), lr=opt.lr, momentum=opt.momentum, weight_decay=opt.weight_decay)
elif opt.optim=="Adam":
	optimizer = optim.Adam(model.parameters(),lr=opt.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=opt.weight_decay)

#define loss
criterion = nn.CrossEntropyLoss()
#define sheduler
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,[10,20,50,70], gamma=opt.gamma, last_epoch=-1)

for epoch in range(opt.epochs):
	lr_scheduler.step()
	train_loss, train_acc = train_test(trainloader, model, criterion,use_cuda,train_flag=True)
	print("%d epoch,train_loss:%f,train_acc:%f"%(epoch,train_loss,train_acc/100.0))
	if (epoch+1)%opt.fre_print==0:
		test_loss, test_acc = train_test(testloader, model, criterion,use_cuda,train_flag=False)
		if test_acc >best_acc:
			best_acc = test_acc
			torch.save({
                	'epoch': epoch + 1,
                	'state_dict': model.state_dict(),
                	'acc': test_acc,
                'best_acc': best_acc,
                'optimizer' : optimizer.state_dict(),
            },os.path.join(opt.save_dir,'model_best.pth'))
		print("%d epoch,test_loss:%f,test_acc:%f,best_acc:%f"%(epoch,test_loss,test_acc/100.0,best_acc))
