import time
import torch
from matplotlib import pyplot as plt

use_cuda = torch.cuda.is_available()
dtype = torch.float32 if use_cuda else torch.float64
device_id = "cuda:0" if use_cuda else "cpu"

def KMeans_cosine(x, K=10, Niter=10, verbose=True):
    """Implements Lloyd's algorithm for the Cosine similarity metric using only PyTorch."""
    
    start = time.time()
    N, D = x.shape  # Number of samples, dimension of the ambient space
    
    # Initialize centroids by selecting the first K points and normalizing them
    c = x[:K, :].clone()
    c = torch.nn.functional.normalize(c, dim=1, p=2)
    
    for i in range(Niter):
        # Compute cosine similarity (dot product of normalized vectors)
        similarities = torch.matmul(x, c.T)  # (N, K)
        
        # Assign points to the nearest centroid
        cl = similarities.argmax(dim=1)
        
        # Recompute centroids: sum points in each cluster
        c.zero_()
        
        # Compute new centroids by summing points assigned to each cluster
        counts = torch.bincount(cl, minlength=K).float().view(K, 1)
        mask = cl[:, None] == torch.arange(K, device=x.device)[None, :]
        new_c = torch.einsum("nd,nk->kd", x, mask.float())
        
        # Avoid division by zero
        c = torch.where(counts > 0, new_c / counts, c)
        
        # Normalize the centroids
        c = torch.nn.functional.normalize(c, dim=1, p=2)
    
    if verbose:
        end = time.time()
        print(f"K-means for cosine similarity with {N:,} points in dimension {D:,}, K = {K:,}:")
        print(f"Timing for {Niter} iterations: {end - start:.5f}s = {Niter} x {(end - start) / Niter:.5f}s\n")
    
    return cl, c