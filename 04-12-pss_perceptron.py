import numpy as np


def perceptron(x, w):
    '''
    x is a vector of inputs and w is a column vector
    '''
    if x @ w > 0:
        return 1
    else:
        return 0


D = np.array([[255, 128, 128, 0, 0], [55, 128, 128, 128, 2], [192, 128, 128, 0, 0],
              [100, 128, 128, 100, 2], [30, 64, 128, 30, 4], [20, 64, 128, 0, 4]])
X = D[:, :4]
y = D[:, -1]
unique_ys = np.unique(y)
X = np.hstack((np.ones((X.shape[0], 1)), X))
X = X/D.max()  # normalization
Y = np.zeros((y.shape[0], y.max()+1))
Y[np.arange(y.shape[0]), y] = 1  # one-hot encoding
w = np.zeros((X.shape[1], Y.shape[1]))

epochs = 3
for epoch in range(epochs):
    print(f'--- epoch {epoch} ---')
    for label in unique_ys:
        predictions = X[:, :] @ w[:, label]
        predictions = np.where(predictions > 0, 1, 0)

        for n,sample in enumerate(X):
            if Y[n, label] != Y[n, label]*perceptron(sample,w[:, label]):
                w[:, label] += (Y[n, label]-Y[n, label]*perceptron(sample, w[:, label])) * sample
            # print(f'y={label},sample_n={n},w=\n{w}')
        scores = X[:, :] @ w[:, label]
        predictions = np.where(scores>0,1,0)
        diff = np.abs(Y[:, label] - predictions)
        accuracy= np.mean(diff)
        print(f'accuracy:{accuracy}')

print('done')
