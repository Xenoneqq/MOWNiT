import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.legendre import legroots
from scipy.interpolate import lagrange, CubicSpline

def chebyshev_nodes(n, a = -1, b = 1):
    return 0.5 * ((b - a) * (-np.cos((2 * np.arange(1, n + 1) - 1) / (2 * n) * np.pi)) + (b + a))

def uniform_nodes(n, a, b):
    return np.linspace(a, b, n)

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
        uniform_pts = uniform_nodes(n, 1, -1)
        uniform_dists = geometric_mean_distance(uniform_pts)
        plt.scatter(uniform_pts, uniform_dists, label=f'Uniform (n={n})', marker='^')
        
        plt.xlabel("x values")
        plt.ylabel("Geometric mean distance")
        plt.title(f"Geometric Mean Distances for n={n}")
        plt.legend()
        plt.grid()
        plt.show()

def f1(x):
    return 1 / (1 + 25 * x**2)

def f2(x):
    return np.exp(np.cos(x))

def runges_function():
    n = 12
    x_unif = uniform_nodes(n, -1, 1)
    x_cheb = chebyshev_nodes(n)
    y_unif = f1(x_unif)
    y_cheb = f1(x_cheb)
    
    poly_unif = lagrange(x_unif, y_unif)
    poly_cheb = lagrange(x_cheb, y_cheb)
    spline = CubicSpline(x_unif, y_unif)
    
    x_dense_unif = uniform_nodes(n * 10, -1, 1)
    x_dense_cheb = chebyshev_nodes(n * 10)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_dense_unif, f1(x_dense_unif), 'k-', label='f1(x)')
    plt.plot(x_dense_unif, poly_unif(x_dense_unif), 'r--', label='Lagrange (równom.)')
    plt.plot(x_dense_cheb, poly_cheb(x_dense_cheb), 'b-.', label='Lagrange (Czebyszewa)')
    plt.plot(x_dense_unif, spline(x_dense_unif), 'g:', label='Spline')
    plt.scatter(x_unif, y_unif, color='black', marker='o', label='Węzły równoległe')
    plt.scatter(x_cheb, y_cheb, color='blue', marker='o', label='Węzły Czebyszewa')
    
    plt.xlabel('x')
    plt.ylabel('f1(x)')
    plt.title('Interpolacja funkcji Rungego')
    plt.legend()
    plt.grid()
    plt.show()

def compute_errors(f, a, b, n_values):
    np.random.seed(42)
    x_test = np.random.uniform(a, b, 500)
    y_test = f(x_test)
    
    errors_lagrange_unif = []
    errors_lagrange_cheb = []
    errors_spline = []
    
    for n in n_values:
        x_unif = uniform_nodes(n, a, b)
        x_cheb = chebyshev_nodes(n, a, b)
        y_unif = f(x_unif)
        y_cheb = f(x_cheb)
        
        poly_unif = lagrange(x_unif, y_unif)
        poly_cheb = lagrange(x_cheb, y_cheb)
        spline = CubicSpline(x_unif, y_unif)
        
        error_lagrange_unif = np.linalg.norm(poly_unif(x_test) - y_test)
        error_lagrange_cheb = np.linalg.norm(poly_cheb(x_test) - y_test)
        error_spline = np.linalg.norm(spline(x_test) - y_test)
        
        errors_lagrange_unif.append(error_lagrange_unif)
        errors_lagrange_cheb.append(error_lagrange_cheb)
        errors_spline.append(error_spline)
    
    return errors_lagrange_unif, errors_lagrange_cheb, errors_spline

def plot_errors(f, a, b, title):
    n_values = np.arange(4, 51)
    errors_lagrange_unif, errors_lagrange_cheb, errors_spline = compute_errors(f, a, b, n_values)
    
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, errors_lagrange_unif, 'r--', label='Lagrange (równom.)')
    plt.plot(n_values, errors_lagrange_cheb, 'b-.', label='Lagrange (Czebyszewa)')
    plt.plot(n_values, errors_spline, 'g:', label='Spline')
    
    plt.xlabel('Liczba węzłów interpolacji (n)')
    plt.ylabel('Norma wektora błędów (log)')
    plt.yscale('log')
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()


plot_errors(f1, -1, 1, 'Błąd interpolacji dla f1(x)')
plot_errors(f2, 0, 2*np.pi, 'Błąd interpolacji dla f2(x)')