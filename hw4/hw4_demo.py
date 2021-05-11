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


def train(model,loader_train,loader_test,Loss,optimizer,epochs,device,num_class,name):
    print('------ start train ------')
    """
    Args:
        model: resnet model
        loader_train: training dataloader
        loader_test: testing dataloader
        Loss: loss function
        optimizer: optimizer
        epochs: number of training epoch
        device: gpu/cpu
        num_class: #target class
        name: model name when saving model
    Returns:
        dataframe: with column 'epoch','acc_train','acc_test'
    """
    df=pd.DataFrame()
    df['epoch']=range(1,epochs+1)
    best_model_wts=None
    best_evaluated_acc=0

    model.to(device)
    acc_train=list()
    acc_test=list()
    for epoch in range(1,epochs+1):
        """
        train
        """
        with torch.set_grad_enabled(True):
            model.train()
            total_loss=0
            correct=0
            for images,targets in loader_train:
                images,targets=images.to(device),targets.to(device,dtype=torch.long)
                predict=model(images)
                loss=Loss(predict,targets)
                total_loss+=loss.item()
                correct+=predict.max(dim=1)[1].eq(targets).sum().item()
                # print('epoch',epoch,' correct=',correct,'/',len(loader_train.dataset))
                """
                update
                """
                optimizer.zero_grad()
                loss.backward()  # bp
                optimizer.step()
            total_loss/=len(loader_train.dataset)
            acc=100.*correct/len(loader_train.dataset)
            acc_train.append(acc)
            print(f'epoch{epoch:>2d} loss:{total_loss:.4f} acc:{acc:.2f}%')
        """
        evaluate
        """
        _,acc=evaluate(model,loader_test,device,num_class)
        acc_test.append(acc)
        # torch.save(model.state_dict(), str('epoch'+str(epoch)+'.pt'))
        # update best_model_wts
        if acc>best_evaluated_acc:
            best_evaluated_acc=acc
            best_model_wts=copy.deepcopy(model.state_dict())

        # save model
        print('save model')
        torch.save(best_model_wts,'models/'+name+'.pt')
        model.load_state_dict(best_model_wts)

    df['acc_train']=acc_train
    df['acc_test']=acc_test



    return df


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

def plot(dataframe1,dataframe2,title):
    """
    Arguments:
        dataframe1: dataframe with 'epoch','acc_train','acc_test' columns of without pretrained weights model
        dataframe2: dataframe with 'epoch','acc_train','acc_test' columns of with pretrained weights model
        title: figure's title
    Returns:
        figure: an figure
    """
    fig=plt.figure()
    for name in dataframe1.columns[1:]:
        plt.plot(range(1,1+len(dataframe1)),name,data=dataframe1,label=name[4:]+'(w/o pretraining)')
    for name in dataframe2.columns[1:]:
        plt.plot(range(1,1+len(dataframe2)),name,data=dataframe2,label=name[4:]+'(with pretraining)')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy(%)')
    plt.title(title)
    plt.legend()
    return fig

def plot_confusion_matrix(confusion_matrix):
    fig, ax = plt.subplots()
    ax.matshow(confusion_matrix, cmap=plt.cm.Blues)
    ax.xaxis.set_label_position('top')
    for i in range(confusion_matrix.shape[0]):
        for j in range(confusion_matrix.shape[1]):
            ax.text(i, j, '{:.2f}'.format(confusion_matrix[j, i]), va='center', ha='center')
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    return fig




if __name__ == '__main__':
    print(torch.__version__)
    print(torch.cuda.is_available())
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_train=RetinopathyLoader(img_path='data',mode='train')
    loader_train=DataLoader(dataset=dataset_train,batch_size=2,shuffle=True,num_workers=4)

    dataset_test=RetinopathyLoader(img_path='data',mode='test')
    loader_test=DataLoader(dataset=dataset_test,batch_size=2,shuffle=False,num_workers=4)
    num_class = 5
    batch_size = 4
    lr = 1e-3
    epochs_fine_tuning = 8
    momentum = 0.9
    weight_decay = 5e-4
    Loss = nn.CrossEntropyLoss(weight=torch.Tensor([1.0,10.565217391304348,4.906175771971497,29.591690544412607,35.55077452667814]).to(device))


    model_demo = ResNet18(num_class=5,pretrained=True)
    model_demo.load_state_dict(torch.load('models/resnet18_with_pretraining.pt'))
    model_demo = model_demo.to(device)
    for i in range(10):
        pass
        confusion_matrix,acc = evaluate(model_demo,loader_test,device,5)
        print('acc = ',acc)

    # # finetuning
    # print('~~~ finetuning ~~~')
    # for param in model_demo.parameters():
    #     param.requires_grad=True
    # optimizer=optim.SGD(model_demo.parameters(),lr=lr,momentum=momentum,weight_decay=weight_decay)
    # train(model_demo,loader_train,loader_test,Loss,optimizer,epochs_fine_tuning,device,num_class,'resnet18_with_pretraining')
