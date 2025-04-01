import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legroots

def chebyshev_nodes(n):
    return np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi)

def geometric_mean_distance(points):
    n = len(points)
    distances = np.zeros(n)
    for i in range(n):
        dists = np.abs(points[i] - np.delete(points, i))
        distances[i] = np.exp(np.mean(np.log(dists)))
    return distances

def plot_geometric_mean_distances():
    ns = [10, 20, 50]
    for n in ns:
        plt.figure(figsize=(10, 5))
        
        # Chebyshev nodes
        chebyshev_pts = chebyshev_nodes(n)
        chebyshev_dists = geometric_mean_distance(chebyshev_pts)
        plt.scatter(chebyshev_pts, chebyshev_dists, label=f'Chebyshev (n={n})', marker='o')
        
        # Legendre nodes
        legendre_pts = legroots(np.polynomial.legendre.Legendre.basis(n).coef)
        legendre_dists = geometric_mean_distance(legendre_pts)
        plt.scatter(legendre_pts, legendre_dists, label=f'Legendre (n={n})', marker='s')
        
        # Uniform nodes
        uniform_pts = np.linspace(-1, 1, n)
        uniform_dists = geometric_mean_distance(uniform_pts)
        plt.scatter(uniform_pts, uniform_dists, label=f'Uniform (n={n})', marker='^')
        
        plt.xlabel("x values")
        plt.ylabel("Geometric mean distance")
        plt.title(f"Geometric Mean Distances for n={n}")
        plt.legend()
        plt.grid()
        plt.show()

plot_geometric_mean_distances()
