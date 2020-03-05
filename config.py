class Config(object):
	train_data='/home/daixiangzi/dataset/dog_cat/shell/train.txt'
	test_data = '/home/daixiangzi/dataset/dog_cat/shell/val.txt'
	save_dir = '/home/project/save_model/'
	gpu_id = '0'
	train_batch = 64
	test_batch = 16
	epochs = 100
	seed = 666
	workers=4
	num_class=2
	fre_print=2
	weight_decay = 1e-4
	lr = 0.003
	optim = "Adam" #SGD,Adam
	gamma = 0.1
