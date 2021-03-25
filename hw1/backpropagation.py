import numpy as np
import random
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.special import expit, logit


num_epochs = 300
lr = 0.001
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
    return np.array(inputs,dtype=np.float128),np.array(labels,dtype=np.float128).reshape(n,1)


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
    return np.array(inputs,dtype=np.float128),np.array(labels,dtype=np.float128).reshape(21,1)


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
        if pred_y[i]>0.5:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()


def sigmoid(x):
    # if x< -99999:return 0
    # if x>  99999:return 1
    return 1.0/(1.0+np.exp(-x))


def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)

def MSE(y,pred_y):
    cost = 0
    for i in range(len(y)):
        cost += (y[i]-pred_y[i])*(y[i]-pred_y[i])
    cost = float(cost/len(y))
    return cost

def derivative_MSE(y,pred_y):
    return -2*(y-pred_y)

def init_parameters(num_of_nodes):
    W = list([])
    for i in range(layers+1):
        tmp_tmp_list = []
        for j in range(num_of_nodes[i]):
            tmp_list = []
            for k in range(num_of_nodes[i+1]):
                tmp_list.append( np.random.uniform(0,1) )
            tmp_tmp_list.append(tmp_list)
        W.append(np.array(tmp_tmp_list,dtype=np.float128))
    return W

def init_parameters_zeros(num_of_nodes):
    W = list([])
    for i in range(layers+1):
        tmp_tmp_list = []
        for j in range(num_of_nodes[i]):
            tmp_list = []
            for k in range(num_of_nodes[i+1]):
                tmp_list.append( 0 )
            tmp_tmp_list.append(tmp_list)
        W.append(np.array(tmp_tmp_list,dtype=np.float128))
    return W

def forward(x,W):
    x,W = x.copy(),W.copy()
    Z = list([])
    A = list([])
    for l in range(layers+1):
        # if not l==0:x = A[-1]
        x = np.dot(x,W[l])
        Z.append(x)
        A.append(np.array([sigmoid(item) for item in x],dtype=np.float128))
        # Z.append(np.dot(x,W[l]))
        # A.append(np.array([float(sigmoid(item)) for item in Z[-1]],dtype=np.float128))


    pred_y = A[-1]
    return Z,A,pred_y

def back(W,Z,A,pred_y,x,y,num_of_nodes):
    W,Z,A,pred_y,x,y = W.copy(),Z.copy(),A.copy(),pred_y.copy(),x.copy(),y.copy()
    dW = init_parameters_zeros(num_of_nodes)
    dZ = list([])

    for k in range(layers,-1,-1):
        if k==layers:
            tmp_dZlist = []
            for i in range(len(y)):
                if y[i]-pred_y[i]>=0:
                    tmp_dZlist.append( -derivative_sigmoid(Z[layers][i])  )
                else:
                    tmp_dZlist.append(  derivative_sigmoid(Z[layers][i])  )
            dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))
        else:
            tmp_dZlist = list([])
            for i in range(len(Z[k])):
                dZtmp = 0
                for j in range(len(dZ[0])):
                    dZtmp += dZ[0][j] * W[k+1][i][j]
                dZtmp = dZtmp*derivative_sigmoid(Z[k][i])
                tmp_dZlist.append(dZtmp)
            dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))

    for i in range(layers+1):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                if i==0:
                    dW[i][j][k] = x[j] * dZ[i][k]
                else:
                    dW[i][j][k] = dZ[i-1][j] * dZ[i][k]
    return dW,dZ

def update_weight(W,dW):
    # print(dW)
    new_W = init_parameters_zeros(num_of_nodes)
    for i in range(layers+1):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                new_W[i][j][k] = W[i][j][k]+lr*dW[i][j][k]

    return new_W




if __name__ == '__main__':
    # init
    N = 100
    x,y = generate_linear(n=N)
    num_of_nodes = [x.shape[1], 5, 5, y.shape[1]]
    W = init_parameters(num_of_nodes)

    loss_list = []
    for epoch in range(300000):
        dW_total = []
        loss_total = []
        pred_y_list = []
        for data_num in range(N):
            Z,A,pred_y = forward(x[data_num],W)
            # print(data_num,'pred_y',pred_y)
            pred_y_list.append(pred_y)
            loss = abs(y[data_num]-pred_y)
            dW,dZ = back(W,Z,A,pred_y,x[data_num],y[data_num],num_of_nodes) #W,Z,A,pred_y,x,y

            dW_total.append(dW)
            loss_total.append(loss)


        # dW_mean
        dW_mean = init_parameters_zeros(num_of_nodes)
        for l in range(len(dW_total)):
            for i in range(layers+1):
                for j in range(len(dW_mean[i])):
                    for k in range(len(dW_mean[i][j])):
                        dW_mean[i][j][k] += dW_total[l][i][j][k]
        for i in range(layers+1):
            for j in range(len(dW_mean[i])):
                for k in range(len(dW_mean[i][j])):
                    dW_mean[i][j][k] = float( dW_mean[i][j][k]/len(dW_total) )

        loss_mean = np.mean(np.array(loss_total), axis=0)
        loss_list.append(loss_mean)

        if epoch%250==0 or loss_mean<0.3:
            now = datetime.now()
            dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
            print(dt_string,' epoch=',epoch,'Loss=',loss_mean)

        if loss_mean<0.3:break
        # print('epoch=',epoch,'Loss=',loss_mean)

        W = update_weight(W,dW)
    #
    show_result(x,y,pred_y_list)
    # print(loss_list)
    plt.plot(loss_list,'--', marker="s")
    plt.show()
    for i in range(len(pred_y_list)):print(pred_y_list[i][0])
