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
    A = list([])
    for k in range(layers+1):
        if k==0:
            tmp = np.dot(W[k],x)
        else:
            tmp = np.dot(W[k],Z[k-1])
        tmp_Zlist = np.array([tmp[i] for i in range(len(tmp))])
        tmp_Alist = np.array([sigmoid(tmp[i]) for i in range(len(tmp))])
        Z.append(tmp_Zlist)
        A.append(tmp_Alist)

    pred_y = A[k-1]
    return Z,A,pred_y

def back(W,Z,y,pred_y):
    dW = list([])
    dZ = list([])

    for k in range(layers,-1,-1):
        if k==layers:
            tmp_dZlist = np.array([derivative_MSE(Z[k][i],pred_y[i]) for i in range(len(Z[k]))])
            dZ.insert(0,tmp_dZlist)
        else:
            tmp_dZlist = list([])
            for i in range(len(Z[k])):
                dZtmp = 0
                for j in range(len(Z[k+1])):
                    # print('kij=',k,i,j)
                    # print('len(Z[k+1])=',len(Z[k+1]))
                    # print('dZ[0][j]=',dZ[0][j])
                    # print('W[k+1][i][j]=',W[k+1][j][i])
                    dZtmp += float(dZ[0][j] * W[k+1][j][i])
                dZtmp = float(dZtmp*derivative_sigmoid(Z[k][i]))
                tmp_dZlist.append(dZtmp)
            dZ.insert(0,np.array(tmp_dZlist))
            # tmp_dZlist = np.array([derivative_sigmoid(Z[k][i]) for i in range(len(Z[k]))])
        # print(tmp_dZlist)

    for i in range(len(W)):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                print(i,j,k)

    return dZ



x,y = generate_linear(n=10)
# print(x[0])

num_of_neruals = [x.shape[1], 3, 3, y.shape[1]]
W = init_parameters(num_of_neruals)

Z,A,pred_y = forward([3,5],W)
# print('W=',W)
# print(W[1])
# print('W2=',W[2])
# print(Z[0])
# print(Z[1])
# print(Z[2])
dZ = back(W,Z,y,pred_y)
# print(dZ)
