import numpy as np

#A = np.array([[0,2],[0,0],[0,0]])
A = np.array([[2,0,0],[0,2,1],[0,1,2],[0,0,0]])

# (a) Reduced SVD of A
U_r, S_r, Vt_r = np.linalg.svd(A,full_matrices=False)

# (b) Full SVD of A
U_f, S_f, Vt_f = np.linalg.svd(A, full_matrices=True)

print(np.matmul(A.T,A))

print("(a) Reduced SVD of A:")
print("U_r:")
print(U_r)
print("S_r:")
print(np.diag(S_r))
print("Vt_r:")
print(Vt_r)

print("\n(b) Full SVD of A:")
print("U_f:")
print(U_f)
print("S_f:")
print(np.diag(S_f))
print("Vt_f:")
print(Vt_f)
