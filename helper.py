import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle

def toeplitz_projection(a):
    n = a.shape[0]
    for i in range(0,n):
        right = np.mean(np.diagonal(a, offset=i))
        np.fill_diagonal(a[:,i:], right)
        left = np.mean(np.diagonal(a, offset=-i))
        np.fill_diagonal(a[i:,:], left)
    return a

def hankel_projection(a):
    n = a.shape[0]
    b = np.zeros(a.shape)
    # For each possible sum s = i+j (ranging from 0 to 2n-2)
    for s in range(0, 2*n-1):
        # Calculate the denominator: n - |s - (n-1)|
        denominator = n - abs(s - (n-1))
        # Sum all elements where k+l = s
        total = 0
        count = 0
        # Iterate through all possible (k,l) pairs where k+l = s
        for k in range(max(0, s-(n-1)), min(n, s+1)):
            l = s - k
            if l >= 0 and l < n:
                total += a[k,l]
                count += 1
        # Calculate the average value for this anti-diagonal
        if count > 0:
            avg = total / denominator
            # Fill all positions where i+j = s with the average
            for i in range(max(0, s-(n-1)), min(n, s+1)):
                j = s - i
                if j >= 0 and j < n:
                    b[i,j] = avg
    return b

def PSD_projection(a):
    ah = (a+a.T)/2
    u, h = polar(ah)
    return (h+ah)/2

#cond_P stands for condition number
def generate_random(n, cond_P):
    log_cond_P = np.log(cond_P)
    s = np.exp((np.random.rand(n)-0.5) * log_cond_P)
    S = np.diag(s)
    U = generate_random_orthogonal(n)
    V = generate_random_orthogonal(n)
    P = U.dot(S).dot(V.T) #using matmul or @ might be preferable here
    return P

def min_1norm(Q,U,VT,sigmas,rho):
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)
    
    #replace the singular values
    for i in range(1,min_dim+1):
        new_sigma = sigmas[i-1] - 1/rho
        if new_sigma > 0:
            S[i-1,i-1] = new_sigma
        else:
            break
    
    return U @ S @VT

def rank(vec):
    for i in range(len(vec)):
        if vec[i] < 1e-12:
            return i
    return len(vec)

def fro_hankel(A,B,C,UB, sb, VBT, rb, UC, sc, VCT, rc):
    p = B.shape[1]
    q = C.shape[0]
    s = sb @ sc.T
    lamda = sb[0]*sb[rb-1]*sc[0]*sc[rc-1]
    mask = np.where(s > 0, np.divide(1,s + lamda * np.divide(1, s, where=s>0)), 0)
    Y = np.random.randn(p,q)
    X = np.random.randn(p,q)
    Z = np.random.randn(p,q)
    
    rel_error = float('inf')  # Initialize relative error
    max_iter = 1000  # Maximum number of iterations as safeguard
    iter_count = 0
    
    while rel_error > 1e-10 and iter_count < max_iter:
        Ym = Y.copy()  # Store previous Y for error calculation
        Y = hankel_projection(X-Z)
        W = Y+Z
        Ap = A - B @ W @ C
        temp = UB.T @ Ap @ VCT.T
        X = temp * mask
        X = VBT.T @ X @ UC.T + W
        Z = Z - X + Y
        rel_error = la.norm(Ym-Y, 'fro')/la.norm(Y, 'fro')
        iter_count += 1
    
    if iter_count == max_iter:
        print(f"Warning: Maximum iterations ({max_iter}) reached in fro_hankel")
    
    return X

def fro_hankel_distance(A,B,C,UB, sb, VBT, rb, UC, sc, VCT, rc, Ex,delta):
    p = B.shape[1]  
    q = C.shape[0]  
    sigmaB = np.zeros(p)
    sigmaC = np.zeros(q)
    sigmaB[:rb] = sb
    sigmaC[:rc] = sc
    s = np.outer(sigmaB, sigmaC)
    lamda = sb[0]*sb[rb-1]*sc[0]*sc[rc-1]
    mask = np.where(s > 0, np.divide(1,s + np.divide(lamda,s)), 0)
    # Initialize all matrices with correct dimensions
    X = np.random.randn(p,q)
    Y = np.random.randn(p,q)
    Z1 = np.random.randn(p,q)
    W = np.random.randn(p,q)
    Z2 = np.random.randn(p,q)
    
    rel_error = float('inf')
    max_iter = 10000
    iter_count = 0
    
    while rel_error > 1e-12 and iter_count < max_iter:
        Ym = Y.copy()
        Y = hankel_projection(W-Z2)
        T = X+Y+0.5*(Z1+Z2)
        direction = T - Ex
        distance = la.norm(direction, 'fro')
        if distance < delta:
            W = T
        else:
            W = Ex + (delta/distance) * direction
        
        P = W-Z1
        Ap = A - B @ P @ C  
        temp = UB.T @ Ap @ VCT.T
        X = np.zeros((p,q))
        X[:rb,:rc] = temp[:rb,:rc]
        X = X * mask
        X = VBT.T @ X @ UC.T + P
        Z1 = Z1 + X - W
        Z2 = Z2 + Y - W
        rel_error = la.norm(Ym-Y, 'fro')/la.norm(Y, 'fro')
        iter_count += 1
    
    if iter_count == max_iter:
        print(f"Warning: Maximum iterations ({max_iter}) reached in fro_hankel")
    
    return X

def nuclear_hankel(A, B, C, rho, iteration = 300):
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = fro_hankel(A + D - Y, B, C, UB, sb, VBT, rb, UC, sc, VCT, rc)
        D = D - Y + A - B @ X @ C
    
    return X


def nuclear_hankel_distance(A, B, C, rho, Ex, delta, iteration = 300):
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = fro_hankel_distance(A,B,C,UB, sb, VBT, rb, UC, sc, VCT, rc, Ex,delta)
        D = D - Y + A - B @ X @ C
    
    return X

def fro_PSD(A,B,C,UB, sb, VBT, rb, UC, sc, VCT, rc):
    p = B.shape[1]
    q = C.shape[0]
    s = sb @ sc.T
    lamda = sb[0]*sb[rb-1]*sc[0]*sc[rc-1]
    mask = np.where(s > 0, np.divide(1,s + lamda * np.divide(1, s, where=s>0)), 0)
    Y = np.random.randn(p,q)
    X = np.random.randn(p,q)
    Z = np.random.randn(p,q)
    
    rel_error = float('inf')  # Initialize relative error
    max_iter = 1000  # Maximum number of iterations as safeguard
    iter_count = 0
    
    while rel_error > 1e-10 and iter_count < max_iter:
        Ym = Y.copy()  # Store previous Y for error calculation
        Y = PSD_projection(X-Z)
        W = Y+Z
        Ap = A - B @ W @ C
        temp = UB.T @ Ap @ VCT.T
        X = np.zeros((p,q))
        X[:rb,:rc] = temp[:rb,:rc]
        X = X * mask
        X = VBT.T @ X @ UC.T + W
        Z = Z - X + Y
        rel_error = la.norm(Ym-Y, 'fro')/la.norm(Y, 'fro')
        iter_count += 1
    
    if iter_count == max_iter:
        print(f"Warning: Maximum iterations ({max_iter}) reached in fro_PSD")
    
    return X

def nuclear_PSD(A, B, C, rho, iteration = 300):
    UB, sb, VBT = la.svd(B, full_matrices=True)
    UC, sc, VCT = la.svd(C, full_matrices=True)
    rb = rank(sb)
    rc = rank(sc)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    iter_count = 0
    rel_error = float('inf')
    
    while rel_error > 1e-12 and iter_count < iteration:
        Ym = Y.copy()
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = fro_PSD(A + D - Y, B, C, UB, sb, VBT, rb, UC, sc, VCT, rc)
        D = D - Y + A - B @ X @ C
        rel_error = la.norm(Ym-Y, 'fro')/la.norm(Y, 'fro')
    return X