import numpy as np

A = np.array([[0,2],[0,0],[0,0]])

# Find the reduced SVD of A
U, s, V = np.linalg.svd(A)
print('U matrix:' ,U)
print('singular values:',s)
print('V matrix:',V)
