import numpy as np
import random
import matplotlib.pyplot as plt

batch_size = 16
num_epochs = 300
lr = 0.0002
layers = 2


def generate_linear(n=100):
    pts = np.random.uniform(0,1,(n,2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0],pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if (pt[0]>pt[1]):
            labels.append(0)
        else :
            labels.append(1)
    return np.array(inputs),np.array(labels).reshape(n,1)


def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i,0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i,-0.1*i])
        labels.append(1)
    return np.array(inputs),np.array(labels).reshape(21,1)


def show_result(x,y,pred_y):
    plt.subplot(1,2,1)
    plt.title('Ground Truth',fontsize=18)
    for i in range(x.shape[0]):
        if y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')

    plt.subplot(1,2,2)
    plt.title('Predict Result',fontsize=18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)

def MSE(y,pred_y):
    cost = 0
    for i in range(len(y)):
        print((y[i],pred_y[i]))
        cost = cost+ (y[i]-pred_y[i])**2
    cost = float(cost/len(y))
    return cost

def derivative_MSE(y,pred_y):
    return -2*(y-pred_y)

def init_parameters(num_of_neruals):
    W = list([])
    for k in range(layers+1):
        if k==0:
            tmp_list = [[1 for n in range(num_of_neruals[k])] for m in range(num_of_neruals[k+1])]
        else:
            tmp_list = [[1 for n in range(num_of_neruals[k])] for m in range(num_of_neruals[k+1])]
        W.append(np.array(tmp_list))
    return W

def forward(x,W):
    Z = list([])
    for k in range(layers+1):
        if k==0:
            tmp = np.dot(W[k],x)
        else:
            tmp = np.dot(W[k],Z[k-1])
        tmp_list = np.array([sigmoid(tmp[i]) for i in range(len(tmp))])
        Z.append(tmp_list)

    pred_y = Z[k-1]
    return Z,pred_y

def back(W,Z,y,pred_y):
    dW = W
    for l in range(len(W)-1,-1,-1):
        cost = MSE(y[0],Z[2])
        print(l)


x,y = generate_linear(n=10)

num_of_neruals = [x.shape[1], 3, 3, y.shape[1]]
W = init_parameters(num_of_neruals)

Z,pred_y = forward(x[0],W)
