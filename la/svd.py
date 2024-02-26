import numpy as np

A = np.array([[2,3],[0,2]])

def svd(A):
    # Compute the SVD of A
    #U, s, V = np.linalg.svd(A)
    U,s,V = None  ,None  ,None 

    AAt = np.dot(A, A.T)
    AtA = np.dot(A.T, A)
    print(AAt)
    print(AtA)

    # Compute the eigenvalues and eigenvectors of AAt
    evalues_AAt, evectors_AAt = np.linalg.eig(AAt)
    print('eig values of AAt:',evalues_AAt)
    print('eig vectors of AAt:',evectors_AAt)
    # Compute the eigenvalues and eigenvectors of AtA

    evalues_AtA, evectors_AtA = np.linalg.eig(AtA)

    print('eig values of AtA:',evalues_AtA)
    print('eig vectors of AtA:',evectors_AtA)

    # Sort the eigenvalues and eigenvectors

    # Compute the singular values
    # Compute the singular vectors

    
        
    return U, s, V


U_comp, s_comp, V_comp = svd(A)
U, s, V = np.linalg.svd(A)
print('U matrix:' ,U) 
print('hand implemented U matrix:', U_comp)
print('singular values:',s)
print('hand implemented s matrix:',s_comp)
print('V matrix:',V)
print('hand implemented V matrix:',V_comp)
