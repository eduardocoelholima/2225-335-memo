import numpy as np


D = np.array([[6, 5, 1],
    [4, 3, 1],
    [5, 6, 1],
    [3, 2, 1],
    [3, 4, -1],
    [2, 3, -1]])
X_with_bias = np.column_stack([np.ones((6,1)),D[:,:-1]])
y = D[:,-1].reshape(-1,1)
beta_ls = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
x_test = np.array([[1,3,3.5]]).T
y_hat_ls = beta_ls.T @ x_test
prediction_ls = np.where(y_hat_ls > 0, 1, -1)
print(f'beta_ls={beta_ls.T}, y_hat_ls={y_hat_ls}, prediction_ls={prediction_ls}')

X_no_bias = D[:, :-1]
lambdas = np.array([[0, 0, 0.7, 1.4, 2.0, 0]]).T
# beta_svm_no_bias = \sum_i lambda_i * y_i * x_i.T @ x
beta_svm_no_bias = (lambdas * y).T @ X_no_bias
x_0 = X_no_bias[2,:]
y_0 = y[2,:]
# since 1 = y*(beta_svm_no_bias@x + beta_svm_0)
beta_svm_0 = (1 - y_0 * beta_svm_no_bias @ x_0) / y_0
beta_svm = np.column_stack([beta_svm_0, beta_svm_no_bias]).T
y_hat_svm = beta_svm.T @ x_test
prediction_svm = np.where(y_hat_svm > 0, 1, -1)
print(f'beta_svm={beta_svm.T}, y_hat_svm={y_hat_svm}, prediction_svm={prediction_svm}')
