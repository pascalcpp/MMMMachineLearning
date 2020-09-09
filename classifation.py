import numpy as np
import pandas as pd
import sys
import csv
import random
import matplotlib.pyplot as plt

# np.random.seed(0)

def _shuffle(X, Y):
    # This function shuffles two equal-length list/array, X and Y, together.
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])

def _normalize(X, train=True, specified_column=None, X_mean=None, X_std=None):
    # This function normalizes specific columns of X.
    # The mean and standard variance of training data will be reused when processing testing data.
    #
    # Arguments:
    #     X: data to be processed
    #     train: 'True' when processing training data, 'False' for testing data
    #     specific_column: indexes of the columns that will be normalized. If 'None', all columns
    #         will be normalized.
    #     X_mean: mean value of training data, used when train = 'False'
    #     X_std: standard deviation of training data, used when train = 'False'
    # Outputs:
    #     X: normalized data
    #     X_mean: computed mean value of training data
    #     X_std: computed standard deviation of training data

    if specified_column == None:
        specified_column = np.arange(X.shape[1])
    if train:
        X_mean = np.mean(X[:, specified_column], 0).reshape(1, -1)
        X_std = np.std(X[:, specified_column], 0).reshape(1, -1)


    # 注意数据类型
    X = X.astype(float)
    X[:, specified_column] = (X[:, specified_column] - X_mean) / (X_std + 1e-8)

    return X, X_mean, X_std

def sigmod(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _predict(X, w):
    # This function returns a truth value prediction for each row of X
    # by rounding the result of logistic regression function.
    return np.round(sigmod(np.dot(X,w))).astype(np.int)


def _accuracy(Y_pred, Y_label):
    # This function calculates prediction accuracy
    acc = 1 - np.mean(np.abs(Y_pred - Y_label))
    return acc

def _cross_entropy_loss(y_pred, Y_label):
    # This function computes the cross entropy.
    #
    # Arguements:
    #     y_pred: probabilistic predictions, float vector
    #     Y_label: ground truth labels, bool vector
    # Output:
    #     cross entropy, scalar
    cross_entropy = -np.dot(Y_label.T, np.log(y_pred)) - np.dot((1 - Y_label).T, np.log(1 - y_pred))

    return cross_entropy

def _gradient(X, Y_label, w):
    # This function computes the gradient of cross entropy loss with respect to weight w and bias b.
    y_pred = _predict(X,w)
    pred_error = Y_label - y_pred
    # w_grad = -np.sum(np.dot(X.T,pred_error), 0)
    w_grad = np.dot(X.transpose(), (sigmod(X.dot(w)) - Y_label))
    return w_grad/X.shape[0]
    # return w_grad


def _train_dev_split(X, Y, dev_ratio = 0.25):
    # This function spilts data into training set and development set.
    train_size = int(len(X) * (1 - dev_ratio))
    return X[:train_size], Y[:train_size], X[train_size:], Y[train_size:]


x_data = pd.read_csv('./X_train.csv', encoding = 'utf8')
y_data = pd.read_csv('./Y_train.csv', encoding='utf8')

x_data = x_data.iloc[:, 1:]
y_data = y_data.iloc[:,1:]


x_data = x_data.to_numpy()
y_data = y_data.to_numpy()

X_train = x_data
Y_train = y_data


dim = 510 + 1
w = np.zeros([dim, 1])

# 此时Xtrain中全为整数 所以数据类型不是float

X_train, X_mean, X_std = _normalize(X_train, train = True)
X_train = np.concatenate((np.ones([X_train.shape[0], 1]), X_train), axis = 1).astype(float)

X_test = pd.read_csv('./X_test.csv',encoding='utf8')
X_test = X_test.iloc[:,1:]

X_test = X_test.to_numpy()
X_test,t1,t2= _normalize(X_test,train=False,X_mean=X_mean,X_std=X_std)

X_test = np.concatenate((np.ones([X_test.shape[0], 1]), X_test), axis = 1).astype(float)




mt = np.zeros([dim, 1])
vt = np.zeros([dim, 1])
alpha = 0.001
b1 = 0.9
b2 = 0.999
eps = 0.00000001
#  77 0.283 0.2640
max_iter = 1000
batch_size = 8
learning_rate = 0.1
step = 1
# Iterative training
adagrad = np.zeros([dim, 1])


dev_ratio = 0.1
X_train, Y_train, X_dev, Y_dev = _train_dev_split(X_train, Y_train, dev_ratio = dev_ratio)
train_size = X_train.shape[0]
train_loss = []
dev_loss = []
train_acc = []
dev_acc = []






for epoch in range(max_iter):
    # Random shuffle at the begging of each epoch
    X_train, Y_train = _shuffle(X_train, Y_train)
    # Mini-batch training
    for idx in range(int(np.floor(train_size / batch_size))):

        # print(idx)
        X = X_train[idx * batch_size:(idx + 1) * batch_size]
        Y = Y_train[idx * batch_size:(idx + 1) * batch_size]
        # Compute the gradient
        gradient = _gradient(X,Y,w)

        # optimized algorithm adam
        # mt = b1 * mt + (1 - b1) * gradient
        # vt = b2 * vt + (1 - b2) * (gradient**2)
        # mtt = mt / (1 - (b1**(step)))
        # vtt = vt / (1 - (b2**(step)))
        w = w - learning_rate / np.sqrt(step) * gradient

        # gradient descent update
        # learning rate decay with time
        # w = w - learning_rate * gradient
        # adagrad += gradient ** 2
        #
        # w = w - learning_rate * gradient/ np.sqrt(adagrad + eps)
        step+=1
        # w = w - alpha * mtt/(np.sqrt(vtt)+eps)



    y_train_pred = sigmod(np.dot(X_train,w))
    Y_train_pred = _predict(X_train,w)
    y_dev_pred = sigmod(np.dot(X_dev,w))
    Y_dev_pred = _predict(X_dev,w)
    print('=========='+str(epoch)+'============')

    print(_accuracy(Y_dev_pred, Y_dev))
    print(_cross_entropy_loss(y_dev_pred, Y_dev)/X_dev.shape[0])


    print(_accuracy(Y_train_pred, Y_train))
    print(_cross_entropy_loss(y_train_pred, Y_train)/X_train.shape[0])

    train_acc.append(float(_accuracy(Y_train_pred, Y_train)))
    train_loss.append(float(_cross_entropy_loss(y_train_pred, Y_train) / X_train.shape[0]))

    dev_acc.append(float(_accuracy(Y_dev_pred, Y_dev)))
    dev_loss.append(float(_cross_entropy_loss(y_dev_pred, Y_dev)/X_dev.shape[0]))
    # print('======================')
    # print(train_acc)
    # print(train_loss)
    # train_acc.append())
    # train_loss.append(_cross_entropy_loss(y_train_pred, Y_train) / train_size)

print('Training loss: {}'.format(train_loss[-1]))
print('Development loss: {}'.format(dev_loss[-1]))
print('Training accuracy: {}'.format(train_acc[-1]))
print('Development accuracy: {}'.format(dev_acc[-1]))

plt.plot(train_loss)
plt.plot(dev_loss)
plt.title('Loss')
plt.legend(['train', 'dev'])
plt.savefig('loss.png')
plt.show()

# Accuracy curve
plt.plot(train_acc)
plt.plot(dev_acc)
plt.title('Accuracy')
plt.legend(['train', 'dev'])
plt.savefig('acc.png')
plt.show()
predict_y = _predict(X_test,w)


with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'label']
    # print(header)
    csv_writer.writerow(header)
    for i in range(X_test.shape[0]):
        row = [str(i), predict_y[i][0]]
        csv_writer.writerow(row)
        # print(row)