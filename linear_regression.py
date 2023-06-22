import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from d2l import torch as d2l
from torch.utils import data
from torch import nn


def synthetic_data(w, b, num_examples):
    # 生成 y = w*x + b
    X = torch.normal(0, 1, (num_examples, len(w)))
    y = torch.matmul(X, w) + b
    # 添加噪声
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape(-1, 1)

def data_iter(batch_size, features, labels):
    #打乱index顺序
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(indices[i:min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]

def linear_reg(X, w, b):
    '''定义回归模型'''
    return torch.matmul(X, w) + b

def squared_loss(y_hat, y):
    '''均方误差损失函数'''
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

def train_lin():
    lr = 0.03
    num_epochs = 3
    net = linear_reg
    loss = squared_loss

    w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    true_w = torch.tensor([3, -5.6])
    true_b = 4.2
    features, labels = synthetic_data(true_w, true_b, 1000)
    # plt.scatter(features[:, 1].numpy(), labels.numpy())
    # plt.show()

    batch_size = 10

    for epoch in range(num_epochs):
        for X, y in data_iter(batch_size, features, labels):
            # print(x, '\n', y)
            # break;
            l = loss(net(X, w, b), y)
            l.sum().backward()
            sgd([w, b], lr, batch_size)
        with torch.no_grad():
            train_l = loss(net(features, w, b), labels)
            print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
    print(f'w训练后的误差：{true_w - w.reshape(true_w.shape)}')
    print(f'b训练后的误差：{true_b - b}')


###################linear regression using nn#####################
def load_array(data_array, batch_size, train = True):
    '''数据迭代器'''
    dataset = data.TensorDataset(*data_array)
    return data.DataLoader(dataset, batch_size, shuffle = train)

def train_lin_nn():
    true_w = torch.tensor([3, -5.6])
    true_b = 4.2
    lr = 0.03
    num_epochs = 3
    batch_size = 10

    features, lables = d2l.synthetic_data(true_w, true_b, 1000)
    data_iter = load_array((features, lables), batch_size)

    net = nn.Sequential(nn.Linear(2, 1))
    net[0].weight.data.normal_(0, 0.01)
    net[0].bias.data.fill_(0)

    loss = nn.MSELoss()
    trainer = torch.optim.SGD(net.parameters(), lr = lr)

    for epoch in range(num_epochs):
        for X, y in data_iter:
            l = loss(net(X), y)
            trainer.zero_grad()
            l.backward()
            trainer.step()
        l = loss(net(features), lables)
        print(f'epoch {epoch + 1}, loss {l:f}')
    
    print(f'w训练后的误差：{true_w - net[0].weight.reshape(true_w.shape)}')
    print(f'b训练后的误差：{true_b - net[0].bias}')


    


if __name__ == '__main__':
    # train_lin()
    train_lin_nn()
        