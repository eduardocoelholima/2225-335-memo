import numpy as np
X = np.array([[6,5],[5,6],[4,3],[3,4],[2,3],[3,2]])
y = np.array([[1],[2],[1],[0],[2],[1]])
X
y
beta = np.linalg.inv(X.T @ X) @ X.T @ y
beta
X @ beta
beta2 = np.linalg.pinv(X) @ y
beta2
X @ beta2
