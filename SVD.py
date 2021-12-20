"""
Morad BEN TAYEB - 2021

Compute low rank approximation of matrix with SVD
Let A be a m x n matrix, then :
    - We first generate a gaussian (or SRHT) n x k matrix Omega
    - Compute Y = A . Omega 
    - Apply QR decomposition to Y
    - Compute B = Qt . A 
    - Compute SVD of B (truncated)
"""

import numpy as np 
import matplotlib.pyplot as plt 
from PIL import Image
import time
from numpy.lib import diag 
from scipy.stats import ortho_group
from scipy.linalg import hadamard

def randSVD(A,k,q=0,choice=0,seed=100):
    """
    Randomized SVD algorithm
        - q is the power to compute (A @ A.T)**q 
        - Choice = 0 mean Omega is N(0,1)
        - Choice = 1 mean Omega is SRHT
    """
    if choice == 0:
        np.random.seed(seed)
        Omega = np.random.randn(A.shape[1], k)

    if choice == 1:
        Omega = SRHT(A.shape[1],k)


    start = time.time()
    Y =  A @ Omega 
    for i in range(q):
        Y = A @ (A.T @ Y)
    Q, _ = np.linalg.qr(Y)
    B = Q.T @ A 
    ut,s,v = np.linalg.svd(B,full_matrices=0)
    u = (Q @ ut)
    
    return(Q,u,s,v,time.time()-start)

def SRHT(n,l):
    """
    Computes a subsampled random Hadamard transform 
    """
    rng = np.random.default_rng()
    d = np.random.choice([-1,1],n)
    D = diag(d)
    H = n**(-1/2)*hadamard(n)
    P = rng.permutation(np.eye(n),axis=1)[:,:l]
    
    return(np.sqrt(n/l) * D @ H @ P)


def svd(A):
    """
    To compute SVD
    """
    return(np.linalg.svd(A,full_matrices=0))


px, py = 4032, 3024 # Original size of the pictures
r = (py/px)

def select(i,N):
    """
    Method to select an image, for convenience
    """
    if i==1:
        im = Image.open("nuit.jpg").resize((N,int(N*r)))
    if i==2:
        im = Image.open("arbres.jpg").resize((N,int(N*r)))
    if i==3:
        im = Image.open("coucher.jpg").resize((N,int(N*r)))
    if i==4:
        im = Image.open("moi.jpg").resize((N,int(N*r)))
    if i==5:
        im = Image.open("montagne_lac.jpg").resize((N,int(N*r)))
    
    return(im)



def test_matrix(case,n):
    """
    Test matrix : 
        - case = 1 : exponential decay of singular values with noise
        - case = 2 : slow decay of singular values
    """
    # Random orthogonal matrix
    U = ortho_group.rvs(dim=n)
    V = ortho_group.rvs(dim=n)
    if case == 1:
        s = [np.exp(-i/7) for i in range(n)]
        A = U @ np.diag(s) @ V
    if case == 2:
        s = [10**(-4*(i)/19) if i in range(20) else 10**(-4)/(i-19)**(1/10) for i in range(n)]
        A = U @ np.diag(s) @ V
    
    return(s,A)


def display(A,A1,A2,k,title=""):
    """
    Displaying both randomized and classic SVD
    """
    plt.subplots(1,3)
    plt.subplot(1,3,1)
    plt.imshow(A)
    plt.axis("off")
    plt.title("Original with dimension {0}x{1}".format(A.shape[0],A.shape[1]))
    
    plt.subplot(1,3,2)
    plt.imshow(A2)
    plt.axis("off")
    plt.title("rank-{0} classic SVD ".format(k))
    
    plt.subplot(1,3,3)
    plt.imshow(A1)
    plt.title("Rank-{0} randomized SVD ".format(k) + title)  
    plt.axis("off")
    plt.show()    
    
    
    return()

def vis_SV(s,label,ln=None):
    """
    For visualization of singular values
    """
    X=range(len(s))
    plt.plot(X,s,label=label,linestyle=ln)
    plt.yscale("log")
    plt.xlabel("k")
    plt.ylabel("Singular values (log scale)")
    plt.legend()

def L2_norm_k(s,k):
    """
    Computes L2 norm from index k
    """
    return(np.sum(s[k:])**0.5)

def compute_error(A,k,q=0,choice=0,step=1):
    """
    Compute norm2(A-QQtA)
    """
    l_err = []
    lk = range(1,k,step)
    for l in lk:
        Q,_,_,_,_ = randSVD(A,l,q=q,choice=choice)
        err = np.linalg.norm((np.eye(A.shape[0]) - Q @ Q.T) @ A, 2)
        l_err += [err]

    return(np.array(lk),np.array(l_err))


"""
MAIN PROGRAM 
"""
if __name__ == "__main__":

    n = 2**10 # Column size
    p = 0 # Oversampling
    k = 50

    im = select(3,n) 
    A = np.asarray(im)[:,:,1]
    # s, A = test_matrix(2,n)
    print("Matrix A ready")
    
    # Computing randomized SVD
    _,u1,s1,v1,exTime = randSVD(A,k+p,q=0,choice = 0)
    print("Randomized SVD", exTime)
    A1 = u1[:,:k] @ np.diag(s1[:k]) @ v1[:k,:]

    # Computing randomized SVD q=3
    _,u1t,s1t,v1t,exTime = randSVD(A,k+p,q=3,choice = 0)
    print("Randomized SVD", exTime)
    A1t = u1t[:,:k] @ np.diag(s1t[:k]) @ v1t[:k,:]

    # Computing SVD
    start = time.time()
    u2,s2,v2 = svd(A)
    print("SVD", time.time() - start)
    A2 = u2[:,:k] @ np.diag(s2[:k]) @ v2[:k,:]


    # Displaying images 
    display(A, A1, A2, k,"q=0")
    display(A, A1t, A2, k,"q=3")
    
    # Diplaying decay of singular values
    plt.figure()
    vis_SV(s1[:k],"Randomized with q=0",ln="-.")
    vis_SV(s1t[:k],"Randomized with q=3",ln="--")
    vis_SV(s2[:k+50],"SVD")
    plt.show()




    
    
    
    
