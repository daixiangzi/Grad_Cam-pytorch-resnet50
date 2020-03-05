# Grad_Cam-pytorch-resnet50
Class activate map 
Train dog and cat dataset by resnet50    
Paper:https://arxiv.org/pdf/1610.02391v1.pdf  
# Install 
python3.6  
pytorch1.2  
python-opencv  
numpy  
cuda 10.0  
#Dataset  
Dog and Cat  
[link](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)  
Pre_model:  
# Usage
##train model
python3 run.py
##generate grad_cam image  
python3  test_grad_cam.py /home/xxx_model.pth /home/your_test_images.jpg  
# Result
![test image](https://github.com/daixiangzi/Caffe-PCN/blob/master/results/test.jpg)  
![crop face](https://github.com/daixiangzi/Caffe-PCN/blob/master/results/crop.jpg)  
