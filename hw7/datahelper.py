from torch.utils.data import Dataset
import os
import json
from PIL import Image
from torchvision import transforms
import torch

def get_test_conditions():
    """
    :return: (#test conditions,#classes) tensors
    """
    with open('dataset/task_1/objects.json','r') as file:
        classes = json.load(file)
    with open('dataset/task_1/test.json','r') as file:
        test_conditions_list=json.load(file)

    labels=torch.zeros(len(test_conditions_list),len(classes))
    for i in range(len(test_conditions_list)):
        for condition in test_conditions_list[i]:
            labels[i,int(classes[condition])]=1.

    return labels

class CLEVRDataset(Dataset):
    def __init__(self):
        """
        :param img_path: file of training images
        :param json_path: train.json
        """
        self.max_objects=0
        with open('dataset/task_1/objects.json','r') as file:
            self.classes = json.load(file)
        self.numclasses=len(self.classes)
        self.img_names=[]
        self.img_conditions=[]
        with open('dataset/task_1/train.json','r') as file:
            dict=json.load(file)
            for img_name,img_condition in dict.items():
                self.img_names.append(img_name)
                self.max_objects=max(self.max_objects,len(img_condition))
                self.img_conditions.append([self.classes[condition] for condition in img_condition])
        self.transformations=transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])


    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        # print(('dataset/task_1/images/' + self.img_names[index]))
        img=Image.open('dataset/task_1/images/' + self.img_names[index]).convert('RGB')
        # img=Image.open(os.path.join(self.img_path,self.img_names[index])).convert('RGB')
        img=self.transformations(img)
        condition=self.int2onehot(self.img_conditions[index])
        return img,condition

    def int2onehot(self,int_list):
        onehot=torch.zeros(self.numclasses)
        for i in int_list:
            onehot[i]=1.
        return onehot
