import numpy as np

a = np.array([[1], [2]])
b = np.array([[3], [4]])

print(f'a = {a}')
print(f'b = {b}')
print(f'b.T = {b.T}')
print(f'a + b = {a+b}')
print(f'a * b = {a*b}')
print(f'a @ b = wrong dimensions')
print(f'a.T @ b = {a.T@b}')
print(f'inv(a.T@b) = {np.linalg.inv(a.T@b)}')
