import torch
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,models
import torch.nn as nn
import torch.optim as optim
import copy
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataloader import RetinopathyLoader


class ResNet18(nn.Module):
    def __init__(self,num_class,pretrained=False):
        """
        Args:
            num_class: #target class
            pretrained:
                True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                False: random initialize weights, and all layer's 'require_grad' is True
        """
        super(ResNet18,self).__init__()
        self.model=models.resnet18(pretrained=pretrained)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad=False
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)

    def forward(self,X):
        out=self.model(X)
        return out

class ResNet50(nn.Module):
    def __init__(self,num_class,pretrained=False):
        """
        Args:
            num_class: #target class
            pretrained:
                True: the model will have pretrained weights, and only the last layer's 'requires_grad' is True(trainable)
                False: random initialize weights, and all layer's 'require_grad' is True
        """
        super(ResNet50,self).__init__()
        self.model=models.resnet50(pretrained=pretrained)
        if pretrained:
            for param in self.model.parameters():
                param.requires_grad=False
        num_neurons=self.model.fc.in_features
        self.model.fc=nn.Linear(num_neurons,num_class)

    def forward(self,X):
        out=self.model(X)
        return out



def evaluate(model,loader_test,device,num_class):
    print('------ start evaluate ------')
    """
    Args:
        model: resnet model
        loader_test: testing dataloader
        device: gpu/cpu
        num_class: #target class
    Returns:
        confusion_matrix: (num_class,num_class) ndarray
        acc: accuracy rate
    """
    confusion_matrix=np.zeros((num_class,num_class))

    with torch.set_grad_enabled(False):
        model.eval()
        correct=0
        for images,targets in loader_test:
            images,targets=images.to(device),targets.to(device,dtype=torch.long)
            predict=model(images)
            predict_class=predict.max(dim=1)[1]
            correct+=predict_class.eq(targets).sum().item()
            for i in range(len(targets)):
                confusion_matrix[int(targets[i])][int(predict_class[i])]+=1
        acc=100.*correct/len(loader_test.dataset)

    # normalize confusion_matrix
    confusion_matrix=confusion_matrix/confusion_matrix.sum(axis=1).reshape(num_class,1)

    return confusion_matrix,acc




if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_test=RetinopathyLoader(img_path='data',mode='test')
    loader_test=DataLoader(dataset=dataset_test,batch_size=4,shuffle=False,num_workers=4)


    model_demo = ResNet18(num_class=5,pretrained=True)
    model_demo.load_state_dict(torch.load('models/resnet18_with_pretraining.pt'))
    model_demo = model_demo.to(device)
    confusion_matrix,acc = evaluate(model_demo,loader_test,device,5)
    print('acc = ',acc)
