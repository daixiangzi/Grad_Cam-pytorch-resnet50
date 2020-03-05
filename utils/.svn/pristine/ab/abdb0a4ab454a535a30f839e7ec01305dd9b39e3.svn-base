import torch
class Label_smoothing(torch.nn.Module):
        def __init__(self,num_classes,eps):
            super(Label_smoothing,self).__init__()
            self.eps = eps
            self.v = self.eps/num_classes
            self.logsoft = torch.nn.LogSoftmax(dim=1)
        def forward(self,inputs,label):#inputs is last layer output,label is real class_label.,loss= (1-esp)*softmax(inputs)+esp/num_classes ,when 
            one_hot = torch.zeros_like(inputs)
            one_hot.fill_(self.v)
            y = label.to(torch.long).view(-1,1)
            one_hot.scatter_(1, y, 1-self.eps+self.v)
            loss = - torch.sum(self.logsoft(inputs) * (one_hot.detach())) / inputs.size(0)
            return loss
            
        
            
