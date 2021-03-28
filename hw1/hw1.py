import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import argparse


def parse_test_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--hidden_units', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--N', type=int, default=100)
    parser.add_argument('--activation_func', type=str, default='sigmoid')
    parser.add_argument('--optimizers', type=str, default='GD')
    parser.add_argument('--task', type=str, default='linear')
    return vars(parser.parse_args())



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
        inputs.append([0.1*i,1-0.1*i])
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
    MID = max(pred_y) * 0.5 + min(pred_y) * 0.5
    for i in range(len(pred_y)):
        if pred_y[i]<MID:
            plt.plot(x[i][0],x[i][1],'ro')
        else:
            plt.plot(x[i][0],x[i][1],'bo')

    # accuracy
    correct = 0
    for i in range(len(pred_y)):
        if y[i]==0 and pred_y[i]<MID : correct +=1
        if y[i]==1 and pred_y[i]>=MID : correct +=1

    accuracy = float(1.0 * correct/len(pred_y))
    print('accuracy : ',accuracy)
    plt.show()




def init_parameters(num_of_nodes):
    layers = len(num_of_nodes)-2
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

def init_parameters_zeros(num_of_nodes):
    layers = len(num_of_nodes)-2
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





class NN():
    def __init__(self,config=None):
        self.task = config['task']
        self.num_epochs = config['epochs']
        self.lr = config['lr']
        self.layers = config['layers']
        self.batch_size = config['batch_size']
        self.N = config['N']
        self.hidden_units = config['hidden_units']
        self.activation_func = config['activation_func']
        self.optimizers = config['optimizers']
        print('NN init done')

    def activation(self,x):
        if self.activation_func == 'sigmoid':
            return 1.0/(1.0+np.exp(-x))
        if self.activation_func == 'None':
            return 1.0 * x
        if self.activation_func == 'tanh':
            return np.tanh(x)
        if self.activation_func == "ReLU":
            return np.maximum(0, x)

    def derivative_activation(self,x):
        if self.activation_func == 'sigmoid':
            return np.multiply(x,1.0-x)
        if self.activation_func == 'None':
            return 1.0
        if self.activation_func == 'tanh':
            return 1.0 - x ** 2
        if self.activation_func == "ReLU":
            return 1.0 * (x > 0)

    def forward(self,x,W):
        Z = list([])
        A = list([])
        for l in range(self.layers+1):
            if not l==0:x = A[-1]
            Z.append( np.dot(x,W[l]) )
            A.append(np.array([self.activation(item) for item in Z[-1] ],dtype=np.float128))

        pred_y = A[-1]
        return Z,A,pred_y

    def back(self,W,A,pred_y,x,y,num_of_nodes):
        dW = init_parameters_zeros(num_of_nodes)
        dZ = list([])

        for k in range(self.layers,-1,-1):
            if k == self.layers :
                tmp_dZlist = []
                for i in range(len(y)):
                    tmp_dZlist.append((y[i]-pred_y[i])*self.derivative_activation(A[self.layers][i]))
                dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))
            else :
                tmp_dZlist = list([])
                for i in range(len(A[k])):
                    dZtmp = 0
                    for j in range(len(dZ[0])):
                        dZtmp += dZ[0][j] * W[k+1][i][j]
                    dZtmp = dZtmp*self.derivative_activation(A[k][i])
                    tmp_dZlist.append(dZtmp)
                dZ.insert(0,np.array(tmp_dZlist,dtype=np.float128))

        for i in range(self.layers+1):
            for j in range(len(W[i])):
                for k in range(len(W[i][j])):
                    if not i==0:
                        dW[i][j][k] = dZ[i-1][j] * dZ[i][k]
                    else:
                        dW[i][j][k] = x[j] * dZ[i][k]
        return dW

    def update_weight(self,W,dW):
        if self.optimizers == 'GD':
            new_W = init_parameters_zeros(self.num_of_nodes)
            for i in range(self.layers+1):
                for j in range(len(W[i])):
                    for k in range(len(W[i][j])):
                        new_W[i][j][k] = W[i][j][k] + self.lr * dW[i][j][k]
            return new_W

        if self.optimizers == 'Momentum':
            global vt_last
            vt = init_parameters_zeros(self.num_of_nodes)
            new_W = init_parameters_zeros(self.num_of_nodes)
            beta = 0.9
            try : _ = vt_last
            except NameError: vt_last = init_parameters_zeros(self.num_of_nodes)

            for i in range(self.layers+1):
                for j in range(len(W[i])):
                    for k in range(len(W[i][j])):
                        vt[i][j][k] = beta * vt_last[i][j][k] + self.lr * dW[i][j][k]
                        new_W[i][j][k] = W[i][j][k] + vt[i][j][k]
                        vt_last[i][j][k] = vt[i][j][k]
            return new_W

        if self.optimizers == 'AdaGrad':
            epsilon = 0.00000001
            n = 0
            for i in range(self.layers+1):
                for j in range(len(W[i])):
                    for k in range(len(W[i][j])):
                        n += dW[i][j][k] * dW[i][j][k]

            new_W = init_parameters_zeros(self.num_of_nodes)
            for i in range(self.layers+1):
                for j in range(len(W[i])):
                    for k in range(len(W[i][j])):
                        new_W[i][j][k] = W[i][j][k] + self.lr * dW[i][j][k] * (n+epsilon)**(-0.5)
            return new_W

        if self.optimizers == 'Adam':
            global vt_old
            global mt_old
            vt = init_parameters_zeros(self.num_of_nodes)
            mt = init_parameters_zeros(self.num_of_nodes)
            vt_hat = init_parameters_zeros(self.num_of_nodes)
            mt_hat = init_parameters_zeros(self.num_of_nodes)
            new_W = init_parameters_zeros(self.num_of_nodes)
            beta = 0.9
            epsilon = 0.00000001

            try : _,__ = vt_old,mt_old
            except NameError:
                vt_old = init_parameters_zeros(self.num_of_nodes)
                mt_old = init_parameters_zeros(self.num_of_nodes)
            for i in range(self.layers+1):
                for j in range(len(W[i])):
                    for k in range(len(W[i][j])):
                        vt[i][j][k] = beta * vt_old[i][j][k] + (1.0 - beta) * dW[i][j][k] * dW[i][j][k]
                        mt[i][j][k] = beta * mt_old[i][j][k] + (1.0 - beta) * dW[i][j][k]

                        vt_hat[i][j][k] = vt[i][j][k]/(1.0 - beta)
                        mt_hat[i][j][k] = mt[i][j][k]/(1.0 - beta)

                        new_W[i][j][k] = W[i][j][k] + self.lr * (mt_hat[i][j][k] / ((vt_hat[i][j][k])**0.5+epsilon) )
            return new_W






    def train(self):
        if self.task == 'linear' : x,y = generate_linear(n=self.N)
        if self.task == 'xor':
            x,y = generate_XOR_easy()
            self.N = 21

        # init W
        self.num_of_nodes = [x.shape[1],y.shape[1]]
        for i in range(self.layers) : self.num_of_nodes.insert(1,self.hidden_units)
        W = init_parameters(self.num_of_nodes)

        loss_list = []
        dW_total, loss_total, pred_y_list, count = [], [], [], 0
        for epoch in range(self.num_epochs):
            pred_y_list = []
            for data_num in range(self.N):
                count = count + 1
                Z,A,pred_y = self.forward(x[data_num],W)
                pred_y_list.append(pred_y)
                loss = abs(y[data_num]-pred_y)
                dW = self.back(W,A,pred_y,x[data_num],y[data_num],self.num_of_nodes) #W,Z,A,pred_y,x,y

                # batch
                dW_total.append(dW)
                loss_total.append(loss)

                if count == self.batch_size:
                    # mean
                    dW_mean = init_parameters_zeros(self.num_of_nodes)
                    for l in range(len(dW_total)):
                        for i in range(self.layers+1):
                            for j in range(len(dW_mean[i])):
                                for k in range(len(dW_mean[i][j])):
                                    dW_mean[i][j][k] += dW_total[l][i][j][k]
                    for i in range(self.layers+1):
                        for j in range(len(dW_mean[i])):
                            for k in range(len(dW_mean[i][j])):
                                dW_mean[i][j][k] = float( dW_mean[i][j][k]/len(dW_total) )
                    loss_mean = np.mean(np.array(loss_total), axis=0)

                    #update
                    W = self.update_weight(W,dW_mean)
                    dW_total, loss_total, count = [], [], 0

            if epoch % 500 == 0:
                now = datetime.now()
                dt_string = now.strftime("%H:%M:%S")
                print(dt_string,' epoch=',epoch,'Loss=',loss_mean[0])
            loss_list.append(loss_mean[0])
            if loss_mean[0] < 0.00001 : break


        plt.plot(loss_list,'--', marker="s")
        plt.title('Learning curve')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.show()
        for i in range(len(pred_y_list)):print(pred_y_list[i][0])
        show_result(x,y,pred_y_list)
        print('epochs:',self.num_epochs ,'lr:',self.lr ,'layers:',self.layers ,'batch_size:',self.batch_size ,'N:',self.N ,'hidden_units:',self.hidden_units ,self.activation_func ,self.optimizers)


if __name__ == '__main__':
    config = parse_test_configs()
    nn = NN(config=config)
    nn.train()
