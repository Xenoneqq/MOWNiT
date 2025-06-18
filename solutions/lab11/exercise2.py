import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)

n_segments = 20
n_points = n_segments + 1
k_obstacles = 50
x0_fixed = np.array([0.0, 0.0])
xn_fixed = np.array([20.0, 20.0])
lambda1 = 1.0
lambda2 = 1.0
epsilon = 1e-13
max_iterations = 400

obstacles = np.random.uniform(0, 20, size=(k_obstacles, 2))

# Oblicza wartość funkcji celu F dla danej ścieżki X.
def objective_function(X, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles):
    X_reshaped = X.reshape(n_segments + 1, 2)

    term1_sum = 0
    for i in range(n_segments + 1):
        for j in range(k_obstacles):
            dist_sq = np.sum((X_reshaped[i] - obstacles[j])**2)
            term1_sum += 1 / (epsilon + dist_sq)
    F1 = lambda1 * term1_sum

    term2_sum = 0
    for i in range(n_segments):
        term2_sum += np.sum((X_reshaped[i+1] - X_reshaped[i])**2)
    F2 = lambda2 * term2_sum

    return F1 + F2

# Oblicza gradient funkcji celu F względem każdego punktu x(i).
def gradient_F(X, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles):
    X_reshaped = X.reshape(n_segments + 1, 2)
    grad = np.zeros_like(X_reshaped)

    for p in range(n_segments + 1):
        grad_F1_p = np.zeros(2)
        for j in range(k_obstacles):
            diff = X_reshaped[p] - obstacles[j]
            dist_sq = np.sum(diff**2)
            grad_F1_p += -2 * diff / ((epsilon + dist_sq)**2)
        grad_F1_p *= lambda1

        grad_F2_p = np.zeros(2)
        if p == 0:
            grad_F2_p = -2 * (X_reshaped[1] - X_reshaped[0])
        elif p == n_segments:
            grad_F2_p = 2 * (X_reshaped[n_segments] - X_reshaped[n_segments-1])
        else:
            grad_F2_p = 2 * (X_reshaped[p] - X_reshaped[p-1]) - 2 * (X_reshaped[p+1] - X_reshaped[p])
        grad_F2_p *= lambda2
        
        grad[p] = grad_F1_p + grad_F2_p

    grad[0, :] = 0.0
    grad[n_segments, :] = 0.0

    return grad.flatten()

# Implementacja metody złotego podziału do minimalizacji funkcji jednowymiarowej.
def golden_section_search(func, a, b, tol=1e-6):
    phi = (1 + np.sqrt(5)) / 2
    resphi = 2 - phi

    x1 = a + resphi * (b - a)
    x2 = b - resphi * (b - a)

    f1 = func(x1)
    f2 = func(x2)

    while abs(b - a) > tol:
        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + resphi * (b - a)
            f1 = func(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - resphi * (b - a)
            f2 = func(x2)
    return (a + b) / 2

# Implementuje algorytm największego spadku z przeszukiwaniem liniowym (złoty podział).
def gradient_descent_with_line_search(initial_path_flat, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles, max_iterations):
    path_X_flat = np.copy(initial_path_flat)
    
    losses = []

    for iteration in range(max_iterations):
        current_F_value = objective_function(path_X_flat, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles)
        losses.append(current_F_value)

        grad = gradient_F(path_X_flat, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles)
        
        direction = -grad
        
        line_search_func = lambda alpha: objective_function(path_X_flat + alpha * direction, obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles)

        alpha_optimal = golden_section_search(line_search_func, 0, 100)

        path_X_flat += alpha_optimal * direction

        path_X_flat[0:2] = x0_fixed
        path_X_flat[-2:] = xn_fixed
        
    return path_X_flat.reshape(n_points, 2), losses

num_runs = 5
results = []
fig, axes = plt.subplots(1, num_runs, figsize=(num_runs * 5, 6))
fig.suptitle("Zoptymalizowane ścieżki robota dla różnych inicjalizacji", fontsize=16)

if num_runs == 1:
    axes = [axes]

for run_idx in range(num_runs):
    print(f"\n--- Uruchomienie {run_idx + 1} ---")
    
    initial_path_interior = np.random.uniform(0, 20, size=(n_points - 2, 2))
    
    initial_path_X = np.zeros((n_points, 2))
    initial_path_X[0] = x0_fixed
    initial_path_X[n_points-1] = xn_fixed
    initial_path_X[1:n_points-1] = initial_path_interior
    
    start_time = time.time()
    optimized_path, losses = gradient_descent_with_line_search(
        initial_path_X.flatten(), obstacles, lambda1, lambda2, epsilon, n_segments, k_obstacles, max_iterations
    )
    end_time = time.time()
    
    results.append({
        "optimized_path": optimized_path,
        "losses": losses,
        "time": end_time - start_time,
        "initial_path": initial_path_X
    })

    print(f"Czas obliczeń: {end_time - start_time:.4f} s")
    print(f"Wartość funkcji F po optymalizacji: {losses[-1]:.4f}")

    ax = axes[run_idx]
    ax.plot(optimized_path[:, 0], optimized_path[:, 1], 'b-o', markersize=3, label="Zoptymalizowana ścieżka")
    ax.plot(obstacles[:, 0], obstacles[:, 1], 'rx', markersize=8, label="Przeszkody")
    ax.plot(x0_fixed[0], x0_fixed[1], 'go', markersize=8, label="Start x(0)")
    ax.plot(xn_fixed[0], xn_fixed[1], 'ko', markersize=8, label="Koniec x(n)")
    ax.set_title(f"Run {run_idx + 1}\nF_final={losses[-1]:.2f}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    if run_idx == 0:
        ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(results[0]["losses"], label="Wartość funkcji F")
plt.yscale("log")
plt.xlabel("Iteracja")
plt.ylabel("Wartość funkcji F (log)")
plt.title("Zbieżność funkcji celu F w zależności od iteracji (dla pierwszej inicjalizacji)")
plt.grid(True)
plt.legend()
plt.show()