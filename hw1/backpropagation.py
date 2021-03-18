import numpy as np
import random
import matplotlib.pyplot as plt

num_epochs = 300
lr = 1
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
        if pred_y[i]==0:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')
    plt.show()


def sigmoid(x):
    return float(1.0/(1.0+np.exp(-x)))

def derivative_sigmoid(x):
    return np.multiply(x,1.0-x)

def MSE(y,pred_y):
    cost = 0
    for i in range(len(y)):
        cost = cost+ (y[i]-pred_y[i])**2
    cost = float(cost/len(y))
    return float(cost)

def derivative_MSE(y,pred_y):
    return float(-2*(y-pred_y))

def init_parameters(num_of_nodes):
    W = list([])
    for i in range(layers+1):
        tmp_tmp_list = []
        for j in range(num_of_nodes[i]):
            tmp_list = []
            for k in range(num_of_nodes[i+1]):
                tmp_list.append( np.random.uniform(-1,1) )
            tmp_tmp_list.append(tmp_list)
        W.append(np.array(tmp_tmp_list,dtype=np.float128))
    return W

def forward(x,W):
    Z = list([])
    A = list([])
    for l in range(layers+1):
        if l==0:
            Z.append(np.dot(x,W[l]))
            A.append(np.array([float(sigmoid(item)) for item in Z[-1]],dtype=np.float128))
        else:
            Z.append(np.dot(A[-1],W[l]))
            A.append(np.array([float(sigmoid(item)) for item in Z[-1]],dtype=np.float128))

    pred_y = A[-1]
    return Z,A,pred_y

def back(W,Z,A,pred_y,x,y):
    # print(W,Z)
    dW = W.copy()
    dZ = list([])

    for k in range(layers,-1,-1):
        if k==layers:
            tmp_dZlist = []
            for i in range(len(y)):
                tmp_dZlist.append( float(derivative_MSE(y[i],pred_y[i])* derivative_sigmoid(Z[layers][i]) ) )
            dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))
        else:
            tmp_dZlist = list([])
            for i in range(len(Z[k])):
                dZtmp = 0
                for j in range(len(dZ[0])):
                    dZtmp += float(dZ[0][j] * W[k+1][i][j])
                dZtmp = float(dZtmp*derivative_sigmoid(Z[k][i]))
                tmp_dZlist.append(dZtmp)
            dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))

    for i in range(layers+1):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                if i==0:
                    dW[i][j][k] = float(x[j] * dZ[i][k])
                else:
                    dW[i][j][k] = float(dZ[i-1][j] * dZ[i][k])

    return dW,dZ

def update_weight(W,dW):
    print(dW)
    new_W = W.copy()
    for i in range(layers+1):
        for j in range(len(W[i])):
            for k in range(len(W[i][j])):
                new_W[i][j][k] = float(W[i][j][k]+lr*dW[i][j][k])

    return new_W




if __name__ == '__main__':
    # init
    N = 10
    x,y = generate_linear(n=N)
    num_of_nodes = [x.shape[1], 5, 5, y.shape[1]]
    W = init_parameters(num_of_nodes)

    for epoch in range(1):
        pred_y_list = []
        for data_num in range(N):

            Z,A,pred_y = forward(x[data_num],W)
            pred_y_list.append(pred_y)
            loss = MSE(y[data_num],pred_y)
            dW,dZ = back(W,Z,A,pred_y,x[data_num],y[data_num]) #W,Z,A,pred_y,x,y
            W = update_weight(W,dW)

            print('epoch=',epoch,'data_num=',data_num,'Loss=',loss)

    show_result(x,y,pred_y_list)
