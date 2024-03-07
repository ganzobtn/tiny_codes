import numpy as np

A = np.array([[2,3],[0,2]])
B = np.array([[-1, 1],[2,-1],[3,-5]])

def svd(matrix):

    AtA = np.dot(matrix.T, matrix)
    AAt = np.dot(matrix, matrix.T)
    
    eigenvalues_AtA, U = np.linalg.eigh(AtA)
    eigenvalues_AAt, Vt = np.linalg.eigh(AAt)
    
    idx_AtA = np.argsort(eigenvalues_AtA)[::-1]
    idx_AAt = np.argsort(eigenvalues_AAt)[::-1]
    
    eigenvalues_AtA = eigenvalues_AtA[idx_AtA]
    eigenvalues_AAt = eigenvalues_AAt[idx_AAt]
    
    U = U[:, idx_AtA]
    Vt = Vt[:, idx_AAt]
    
    sigma = np.sqrt(np.abs(eigenvalues_AtA))
    rank = np.sum(eigenvalues_AtA > 0)
    
    U = U[:, :rank]
    Vt = Vt[:, :rank]
    
    V = Vt.T
    S = np.diag(sigma)
    
    return U, S, V


print('--------SVD--------')
U_comp, s_comp, V_comp = svd(A)
U, s, V = np.linalg.svd(A,full_matrices=False)
print('U matrix:')
print(U) 
print('hand implemented U matrix:')
print(U_comp)
print('-------------------')
print('singular values:')
print(s)
print('hand implemented s matrix:')
print(s_comp)
print('-------------------')
print('V matrix:')
print(V)
print('hand implemented V matrix:')
print(V_comp)


print('--------SVD--------')
U_comp, s_comp, V_comp = svd(B)
U, s, V = np.linalg.svd(B,full_matrices=False)
print('U matrix:')
print(U) 
print('hand implemented U matrix:')
print(U_comp)
print('-------------------')
print('singular values:')
print(s)
print('hand implemented s matrix:')
print(s_comp)
print('-------------------')
print('V matrix:')
print(V)
print('hand implemented V matrix:')
print(V_comp)
