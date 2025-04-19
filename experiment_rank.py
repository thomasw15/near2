import timeit
import cvxpy as cp
import numpy as np
from scipy.linalg import polar
import numpy.linalg as la
import matplotlib.pyplot as plt
import pickle

# ============= Helper Functions =============
def find_k(sigmas, rho):
    """Find k for spectral norm minimization"""
    for i in range(1, len(sigmas) + 1):
        if sigmas[i - 1] >= (np.sum(sigmas[:i])-1/rho)/i:
            k = i
    return k

def generate_random_orthogonal(n):
    U, _ = la.qr((np.random.rand(n, n) - 5.) * 200)
    return U

def generate_random(n, cond_P):
    """Generate random matrix with given condition number"""
    log_cond_P = np.log(cond_P)
    s = np.exp((np.random.rand(n)-0.5) * log_cond_P)
    S = np.diag(s)
    U = generate_random_orthogonal(n)
    V = generate_random_orthogonal(n)
    return U @ S @ V.T

def min_1norm(Q, U, VT, sigmas, rho):
    """Minimize nuclear norm"""
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)
    
    for i in range(1, min_dim+1):
        new_sigma = sigmas[i-1] - 1/rho
        if new_sigma > 0:
            S[i-1,i-1] = new_sigma
        else:
            break
    
    return U @ S @ VT

def min_infnorm(Q, U, VT, sigmas, rho, k):
    """Minimize spectral norm"""
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)
    total = np.sum(sigmas)
    
    if total > 1/rho:
        S[:min_dim, :min_dim] = np.diag(sigmas)
        new_sigma = (np.sum(sigmas[:k])-1/rho)/k
        np.fill_diagonal(S[:k,:k], new_sigma)
    
    return U @ S @ VT

def min_3norm(Q, U, VT, sigmas, rho, k):
    """Minimize Schatten 3-norm"""
    S = np.zeros(Q.shape)
    min_dim = min(Q.shape)    
    for i in range(0,min_dim):
        S[i,i] = (-rho + np.sqrt(rho**2 + 12*sigmas[i]*rho))/6
    return U @ S @ VT

def rank_r_approximation(A,r):
    U, s, VT = la.svd(A)
    approx = U[:, :r] @ np.diag(s[:r]) @ VT[:r, :]
    
    return approx

def rank_Frobenius(A, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r):
    P = UB.transpose() @ A @ VCT.transpose()
    P11 = rank_r_approximation(P[:rB, :rC],r)
    X = np.zeros([B.shape[1],C.shape[0]])
    X[:rB, :rC] = np.diag(np.reciprocal(sB[0:rB])) @ P11 @ np.diag(np.reciprocal(sC[0:rC]))
    X = VBT.transpose() @ X @ UC.transpose()
    
    return X

# ============= Solver Functions =============
# solve rank-constrained approximation

def solve_1rank(A, B, C, X_true, r, rho, iteration = 300):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        # X-update
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list
        
def solve_infrank(A, B, C, X_true, r, rho, iteration = 300):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_infnorm(Q,U,VT,sq,rho,k)
        # X-update
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)    
    return X, forward_error_list

def solve_3rank(A, B, C, X_true, r, rho, iteration = 300):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    
    forward_error_list = []
    for i in range(iteration):
        # Y-update
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_3norm(Q,U,VT,sq,rho,k)
        # X-update
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)    
    return X, forward_error_list

def tol_1rank(A, B, C, X_true, r, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        Y = min_1norm(Q,U,VT,sq,rho)
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X
        
def tol_infrank(A, B, C, X_true, r, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_infnorm(Q,U,VT,sq,rho,k)
        # X-update
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X

def tol_3rank(A, B, C, X_true, r, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_3norm(Q,U,VT,sq,rho,k)
        # X-update
        X = rank_Frobenius(A + D - Y, B, C, UB, sB, VBT, rB, UC, sC, VCT, rC, r)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X

# ============= Experiment Functions =============
def run_accuracy_experiment(case=20):
    """Run accuracy comparison experiment"""
    inf_list = []
    one_list = []
    three_list = []
    n_list = list(range(1, case+1))
    
    for i in range(case):
        print(f"Running case {i+1}/{case}")
        n = 32
        B = generate_random(n, max(1,np.random.rand()*n))
        C = generate_random(n, max(1,np.random.rand()*n))
        X_true = generate_random(n, max(1,np.random.rand()*n*n))
        X_true = rank_r_approximation(X_true, 10)
        A = B @ X_true @ C
        
        Xinf, inf_list_temp = solve_infrank(A, B, C, X_true, r=10, rho=0.01, iteration=300)
        X1, one_list_temp = solve_1rank(A, B, C, X_true, r=10, rho=0.01, iteration=300)
        X3, three_list_temp = solve_3rank(A, B, C, X_true, r=10, rho=0.01, iteration=300)

        inf_list.append(inf_list_temp[299])
        one_list.append(one_list_temp[299])
        three_list.append(three_list_temp[299])
    # Save data
    np.savez('accuracy_data_rank.npz',
             n_list=np.array(n_list),
             inf_list=np.array(inf_list),
             one_list=np.array(one_list),
             three_list=np.array(three_list))
    return n_list, inf_list, one_list

def plot_accuracy_results_nuc(data_file='accuracy_data_rank.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    one_list = data['one_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, one_list, 'b-o', label='nuclear norm')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('rank_acc_nuc.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_results_inf(data_file='accuracy_data_rank.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    inf_list = data['inf_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, inf_list, 'b-o', label='spectral norm')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('rank_acc_inf.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_results_3rank(data_file='accuracy_data_rank.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    three_list = data['three_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, three_list, 'b-o', label='Schatten $3$-norm')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('rank_acc_3.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def run_speed_experiment():
    """Run speed comparison experiment"""
    # Dimensions for our algorithms (up to 2^10)
    dimensions_ours = [2**(i+1) for i in range(10)]
    nuclear_times = []
    spectral_times = []
    three_times = []
    #run our algorithms for full range
    for n in dimensions_ours:
        print(f"Testing our algorithms for dimension {n}")
        nuclear_temp = 0
        spectral_temp = 0
        three_temp = 0
        for j in range(10):
            print(f"Testing our algorithms for dimension {n}, trial {j+1}")
            B = generate_random(n, max(1,np.random.rand()*n))
            C = generate_random(n, max(1,np.random.rand()*n))
            X_true = generate_random(n, max(1,np.random.rand()*n*n))
            X_true = rank_r_approximation(X_true, n//2)
            A = B @ X_true @ C
            
            # Nuclear norm timing
            start = timeit.default_timer()
            _ = tol_1rank(A, B, C, X_true, r=n//2, rho=0.0000001, tol=1e-8)
            nuclear_temp += timeit.default_timer() - start
            
            # Spectral norm timing
            start = timeit.default_timer()
            _ = tol_infrank(A, B, C, X_true, r=n//2, rho=0.0000001, tol=1e-8)
            spectral_temp += timeit.default_timer() - start

            # Schatten 3-norm timing
            start = timeit.default_timer()
            _ = tol_3rank(A, B, C, X_true, r=n//2, rho=0.0000001, tol=1e-8)
            three_temp += timeit.default_timer() - start
        
        nuclear_times.append(nuclear_temp/10)
        spectral_times.append(spectral_temp/10)
        three_times.append(three_temp/10)
    # Save data
    np.savez('speed_data_rank.npz',
             dimensions_ours=np.array(dimensions_ours),
             nuclear_times=np.array(nuclear_times),
             spectral_times=np.array(spectral_times),
             three_times=np.array(three_times))
    
    return dimensions_ours, nuclear_times, spectral_times

def plot_speed_results(data_file='speed_data_rank.npz'):
    """Plot speed results from saved data"""
    data = np.load(data_file)
    dimensions_ours = data['dimensions_ours']
    nuclear_times = data['nuclear_times']
    spectral_times = data['spectral_times']
    three_times = data['three_times']
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_ours, nuclear_times, 'b-o', label='nuclear norm')
    plt.plot(dimensions_ours, spectral_times, 'g-o', label='spectral norm')
    plt.plot(dimensions_ours, three_times, 'r-o', label='Schatten $3$-norm')
    plt.xlabel('dimension')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.savefig('speed_comparison_rank.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run experiments and save data
    print("Running accuracy experiment...") 
    run_accuracy_experiment(case=20)
    print("\nGenerating accuracy plots...")
    plot_accuracy_results_nuc()
    plot_accuracy_results_inf()
    plot_accuracy_results_3rank()
    print("\nRunning speed experiment...")
    run_speed_experiment()
    
    # Generate plots from saved data
    print("\nGenerating speed plots...")
    plot_speed_results()
