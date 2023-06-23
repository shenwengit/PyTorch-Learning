
import torch
import torchvision
import matplotlib.pyplot as plt
from torch import nn
from IPython import display
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l


def load_data_fashion_mnist():
    trans = transforms.ToTensor()
    mnist_train = torchvision.datasets.FashionMNIST(root="../data", train=True, transform=trans, download=False)
    mnist_test = torchvision.datasets.FashionMNIST(root="../data", train=False, transform=trans, download=False)
    print(len(mnist_train), len(mnist_test))
    print(mnist_train[0][0].shape)
    return (mnist_train, mnist_test)

def get_fashion_mnist_labels(labels):
    '''获取fashion_mnist对应的标签'''
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    '''show list of images'''
    figsize = (num_rows * scale, num_cols * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy())
        else:
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

def show_a_batch_images(mnist_data):
    X, y = next(iter(data.DataLoader(mnist_data, batch_size=18)))
    show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
    plt.show()

def get_dataloader_workers():
    return 4

def data_to_iter(mnist_data, batch_size):
    return data.DataLoader(mnist_data, batch_size, shuffle=True, num_workers=get_dataloader_workers())


def accuracy(y_hat, y):
    '''计算预测正确的个数'''
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def eval_accuracy(net, data_iter):
    '''计算模型在指定数据上的精度'''
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
        return metric[0] / metric[1]
    
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

def predict(net, test_iter, n=6):  #@save
    """预测标签（定义见第3章）"""
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
    show_images(X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])
    plt.show()

if __name__ == '__main__':
    # show_a_batch_images()

    mnist_train, mnist_test = load_data_fashion_mnist()
    batch_size = 256
    train_iter = data_to_iter(mnist_train, batch_size)
    test_iter = data_to_iter(mnist_test, batch_size)

    net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))
    net.apply(init_weights)

    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=0.1)

    num_epochs = 10
    timer = d2l.Timer()
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)
    print(f'{timer.stop():.2f} sec')
    predict(net, test_iter, 18)



