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

def prod_Frobenius(A, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC):
    FB = F @ VBT.transpose() @ np.diag(np.reciprocal(sB[0:rB]))
    GC = np.diag(np.reciprocal(sC[0:rC])) @ UC.transpose() @ G
    Ahat = UB.transpose() @ A @ VCT.transpose()
    middle = FB.transpose() @ la.inv(FB @ FB.transpose()) @ (H - FB @ Ahat @ GC) @ la.inv(GC.transpose() @ GC) @ GC.transpose()
    middle = middle + Ahat 
    X = VBT.transpose() @ np.diag(np.reciprocal(sB[0:rB])) @ middle @ np.diag(np.reciprocal(sC[0:rC])) @ UC.transpose()
    return X

def solve_3unconstrained(A, B, C, X_true, rho, iteration = 300):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
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
        X = Bpinv @ (A+D-Y) @Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list

def tol_3unconstrained(A, B, C, X_true, rho, tol=1e-8):
    Bpinv = la.pinv(B)
    Cpinv = la.pinv(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        # Y-update
        Q = A + D - B @ X @ C 
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_3norm(Q,U,VT,sq,rho,k)
        # X-update
        X = Bpinv @ (A+D-Y) @Cpinv
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
    return X

def solve_3prod(A, B, C, F, G, H, X_true, rho, iteration = 300):
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
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        forward_error_list.append(forward_error)
    return X, forward_error_list

def tol_3prod(A, B, C, F, G, H, X_true, n, rho, tol=1e-8):
    UB, sB, VBT = la.svd(B)
    rB = la.matrix_rank(B)
    UC, sC, VCT = la.svd(C)
    rC = la.matrix_rank(C)
    Y = np.zeros(A.shape)
    D = np.zeros(A.shape)
    X = np.zeros([B.shape[1],C.shape[0]])
    forward_error = 1
    while forward_error > tol:
        # Y-update
        Q = A + D - B @ X @ C
        U, sq, VT = la.svd(Q)
        k = find_k(sq, rho)
        Y = min_3norm(Q,U,VT,sq,rho,k)
        # X-update
        X = prod_Frobenius(A + D - Y, B, C, F, G, H, UB, sB, VBT, rB, UC, sC, VCT, rC)
        D = D - Y + A - B @ X @ C
        forward_error = la.norm(X - X_true, "fro")/la.norm(X_true, "fro")
        
    return X

def run_accuracy_experiment_3_prod(case=20):
    """Run accuracy comparison experiment"""
    three_list = []
    n_list = list(range(1, case+1))
    
    for i in range(case):
        print(f"Running case {i+1}/{case}")
        n = 32
        B = generate_random(n, max(1,np.random.rand()*n))
        C = generate_random(n, max(1,np.random.rand()*n))
        F = generate_random(n, max(1,np.random.rand()*n))
        G = generate_random(n, max(1,np.random.rand()*n))
        X_true = generate_random(n, max(1,np.random.rand()*n*n))
        H = F @ X_true @ G
        A = B @ X_true @ C
        
        X3, three_list_temp = solve_3prod(A, B, C, F, G, H, X_true, rho=0.01)

        three_list.append(three_list_temp[299])

    # Save data
    np.savez('accuracy_data_prod_3.npz',
             n_list=np.array(n_list),
             three_list=np.array(three_list))
    
    return n_list, three_list

def run_accuracy_experiment_3_unconstrained(case=20):
    """Run accuracy comparison experiment"""
    three_list = []
    n_list = list(range(1, case+1))
    
    for i in range(case):
        print(f"Running case {i+1}/{case}")
        n = 32
        B = generate_random(n, max(1,np.random.rand()*n))
        C = generate_random(n, max(1,np.random.rand()*n))
        X_true = generate_random(n, max(1,np.random.rand()*n*n))
        A = B @ X_true @ C
        
        X3, three_list_temp = solve_3unconstrained(A, B, C, X_true, rho=0.01)
        
        three_list.append(three_list_temp[299])

    
    # Save data
    np.savez('accuracy_data_3.npz',
             n_list=np.array(n_list),
             three_list=np.array(three_list))
    
    return n_list, three_list

def plot_accuracy_results_3prod(data_file='accuracy_data_prod_3.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    three_list = data['three_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, three_list, 'b-o', label='Schatten $3$-norm')
    plt.title('accuracy for Schatten $3$-norms (product constraint)')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('prod_acc_3.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_accuracy_results_3unconstrained(data_file='accuracy_data_3.npz'):
    """Plot accuracy results from saved data"""
    data = np.load(data_file)
    n_list = data['n_list']
    three_list = data['three_list']
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_list, three_list, 'b-o', label='Schatten $3$-norm')
    plt.title('accuracy for Schatten $3$-norms (unconstrained)')
    plt.xlabel('trial')
    plt.ylabel('forward error')
    plt.grid(True)
    plt.xticks(n_list)
    plt.yscale('log')
    plt.legend()
    plt.savefig('unconstrained_acc_3.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def run_speed_experiment_3_unconstrained():
    """Run speed comparison experiment"""
    # Dimensions for our algorithms (up to 2^10)
    dimensions_ours = [2**(i+1) for i in range(10)]
    three_times = []
    
    #run our algorithms for full range
    for n in dimensions_ours:
        print(f"Testing our algorithms for dimension {n}")
        three_temp = 0
        
        for _ in range(10):
            B = generate_random(n, max(1,np.random.rand()*n))
            C = generate_random(n, max(1,np.random.rand()*n))
            X_true = generate_random(n, max(1,np.random.rand()*n*n))
            A = B @ X_true @ C
            
            # Nuclear norm timing
            start = timeit.default_timer()
            _ = tol_3unconstrained(A, B, C, X_true, rho=0.0000001, tol=1e-8)
            three_temp += timeit.default_timer() - start
            
        
        three_times.append(three_temp/10)
    
    # Save data
    np.savez('speed_data_3_unconstrained.npz',
             dimensions_ours=np.array(dimensions_ours),
             three_times=np.array(three_times))
    
    return dimensions_ours, three_times

def run_speed_experiment_3_prod():
    """Run speed comparison experiment"""
    # Dimensions for our algorithms (up to 2^10)
    dimensions_ours = [2**(i+1) for i in range(10)]
    three_times = []
    
    #run our algorithms for full range
    for n in dimensions_ours:
        print(f"Testing our algorithms for dimension {n}")
        three_temp = 0
        
        for _ in range(10):
            B = generate_random(n, max(1,np.random.rand()*n))
            C = generate_random(n, max(1,np.random.rand()*n))
            F = generate_random(n, max(1,np.random.rand()*n))
            G = generate_random(n, max(1,np.random.rand()*n))
            X_true = generate_random(n, max(1,np.random.rand()*n*n))
            H = F @ X_true @ G
            A = B @ X_true @ C
            
            # 3 norm timing
            start = timeit.default_timer()
            _ = tol_3prod(A, B, C, F, G, H, X_true, n, rho=0.0000001, tol=1e-8)
            three_temp += timeit.default_timer() - start
            
        three_times.append(three_temp/10)
    
    # Save data
    np.savez('speed_data_3_prod.npz',
             dimensions_ours=np.array(dimensions_ours),
             three_times=np.array(three_times))
    
    return dimensions_ours, three_times

def plot_speed_results_3_unconstrained(data_file='speed_data_3_unconstrained.npz'):
    """Plot speed results from saved data"""
    data = np.load(data_file)
    dimensions_ours = data['dimensions_ours']
    three_times = data['three_times']
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_ours, three_times, 'b-o', label='Schatten $3$-norm')
    plt.title('speed for Schatten $3$-norms (unconstrained)')
    plt.xlabel('dimension')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.savefig('speed_comparison_3_unconstrained.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_speed_results_3_prod(data_file='speed_data_3_prod.npz'):
    """Plot speed results from saved data"""
    data = np.load(data_file)
    dimensions_ours = data['dimensions_ours']
    three_times = data['three_times']
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions_ours, three_times, 'b-o', label='Schatten $3$-norm')
    plt.title('speed for Schatten $3$-norms (product constraint)')
    plt.xlabel('dimension')
    plt.ylabel('time (s)')
    plt.grid(True)
    plt.xscale('log', base=2)
    plt.yscale('log')
    plt.legend()
    plt.savefig('speed_comparison_3_prod.pdf', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    # Run experiments and save data
    print("Running accuracy experiment...") 
    run_accuracy_experiment_3_unconstrained(case=20)
    run_accuracy_experiment_3_prod(case=20)
    print("\nGenerating accuracy plots...")
    plot_accuracy_results_3unconstrained()
    plot_accuracy_results_3prod()
    print("\nRunning speed experiment...")
    run_speed_experiment_3_unconstrained()
    run_speed_experiment_3_prod()
    # Generate plots from saved data
    print("\nGenerating speed plots...")
    plot_speed_results_3_unconstrained()
    plot_speed_results_3_prod()