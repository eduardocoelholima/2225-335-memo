import numpy as np
from scipy.spatial.distance import mahalanobis
from scipy.stats import multivariate_normal
import math

D = np.array([[3,3,1],[1,1,1],[-1,0,1],[2,2,0],[-2,2,0],[-2,-2,0],[0,-2,0]])
X = D[np.where(D[:,2] == 1)][:,:2]
x_0 = X[0,:]
mu = np.mean(X, axis=0)
sigma = np.cov(X.T)
rank = sigma.shape[0]
d_m = mahalanobis(x_0,mu,np.linalg.inv(sigma))
# d_m = np.sqrt((X[0,:]-mu) @ np.linalg.inv(sigma) @ (X[0,:]-mu).T)
d_m_inv = np.linalg.inv(sigma)
pd = multivariate_normal(mu,sigma).pdf(x_0)
# pd = math.exp(-0.5 * math.pow(d_m,2)) / (math.sqrt(math.pow(2 * math.pi, rank) * np.linalg.det(sigma) ) )
print(f'X={X}\nmu={mu}\nsigma={sigma}\nd_m(x_0,mu,sigma)={d_m}\nd_m^{-1}={d_m_inv}\npd={pd}')
print(f'det(sigma)={np.linalg.det(sigma)}')
