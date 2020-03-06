# Grad_Cam-pytorch-resnet50
Class activate map 
Train dog and cat dataset by resnet50    
Paper:https://arxiv.org/pdf/1610.02391v1.pdf  
# Install 
```bash
python3.6  
pytorch1.2   
cuda 10.0  
```
## Dataset  
Dog and Cat  
[Dataset download link](https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip)  
[Pre_model link](https://pan.baidu.com/s/1nUE9_5fVXqRcp-mhDx6KKw) `Extraction code`:0uqz    
# Usage
## train model
```python
python3 run.py
```
## generate grad_cam image  
```python
python3  test_grad_cam.py /home/xxx_model.pth /home/your_test_images.jpg  
```
# Result
![both](https://github.com/daixiangzi/Grad_Cam-pytorch-resnet50/tree/master/example/both.png) ![dog](https://github.com/daixiangzi/Grad_Cam-pytorch-resnet50/tree/master/example/dog_last.jpg) ![cat](https://github.com/daixiangzi/Grad_Cam-pytorch-resnet50/tree/master/example/cat_last.jpg) 
