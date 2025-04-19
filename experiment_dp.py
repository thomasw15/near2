import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from helper import nuclear_PSD, PSD_projection, fro_PSD

def generate_random_PSD(n):
    """Generate a random PSD matrix"""
    A = np.random.randn(n, n)
    return A @ A.T

def detection_potential(A, BXBt):
    """Compute detection potential between A and BXB^T"""
    # For A
    sA = la.svd(A, full_matrices=False, compute_uv=False)
    dp_A = sA[0] - sA[-1]
    
    # For BXB^T
    sBXBt = la.svd(BXBt, full_matrices=False, compute_uv=False)
    dp_BXBt = sBXBt[0] - sBXBt[-1]
    
    return dp_A / dp_BXBt

def run_experiment():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    n = 20  
    s_values = range(1, 21, 1)  
    rho = 0.01  # penalty parameter for nuclear norm
    
    # Store results
    detection_potentials = []
    
    for s in s_values:
        print(f"\nRunning experiment for s = {s}")
        detection_potentials_temp = []
        for k in range(10):
            print(f"iteration {k}")
            # Generate random PSD matrix A
            A = generate_random_PSD(n)
            # Generate random B
            B = np.random.randn(n, s)
            sb = np.random.rand(s)+1
            B = np.random.randn(n, s)  # Full rank matrix of size n x r
            UB, singular_values, VBT = la.svd(B, full_matrices=True)
            B = UB[:,:s] @ np.diag(sb) @ VBT[:s,:]
            C = B.T  # C = B^T
            
            # Run nuclear_PSD to solve the problem
            X = nuclear_PSD(A, B, C, rho)
            
            # Compute BXB^T
            BXBt = B @ X @ B.T
            
            # Compute detection potential
            dp = detection_potential(A, BXBt)
            detection_potentials_temp.append(dp)
        detection_potentials.append(np.mean(detection_potentials_temp))
        print(f"Detection Potential: {dp:.6f}")
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(list(s_values), detection_potentials, 'b-o')
    plt.xlabel('$s$')
    plt.ylabel('detection potential')
    plt.grid(True)
    plt.xticks(list(s_values))  # Force integer ticks
    plt.savefig('detection_potential.pdf', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save numerical results
    np.savez('experiment_results.npz',
             s_values=np.array(s_values),
             detection_potentials=np.array(detection_potentials))
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"Minimum DP: {min(detection_potentials):.6f} (s = {s_values[np.argmin(detection_potentials)]})")
    print(f"Maximum DP: {max(detection_potentials):.6f} (s = {s_values[np.argmax(detection_potentials)]})")
    print(f"Mean DP: {np.mean(detection_potentials):.6f}")
    print(f"Std DP: {np.std(detection_potentials):.6f}")

if __name__ == "__main__":
    run_experiment() 