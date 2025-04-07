import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial.chebyshev import Chebyshev
from math import inf

# Data: Population of the USA
x = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]
y = [76000000, 92000000, 106000000, 123000000, 132000000,
     151000000, 179000000, 203000000, 227000000]

# Data: Population of USA in 1990
predict_year = 1990
true_value = 248709873

def calculate_product(x, k):
    return sum([x_val ** k for x_val in x])

def calculate_result(x, y, k):
    return sum([x[i]**k * y[i] for i in range(len(x))])

def calculate_matrix_S(x, m):
    matrix = np.zeros((m + 1, m + 1))
    for i in range(m + 1):
        for j in range(m + 1):
            matrix[i][j] = calculate_product(x, i + j)
    return matrix

def calculate_matrix_T(x, y, m):
    matrix = np.zeros((m + 1, 1))
    for i in range(m + 1):
        matrix[i][0] = calculate_result(x, y, i)
    return matrix

def predict(c, x_val):
    return sum([c[i][0] * x_val**i for i in range(len(c))])


best_error = inf
best_aicc = inf 

for m in range(7):
    S = calculate_matrix_S(x, m)
    T = calculate_matrix_T(x, y, m)
    c = np.linalg.solve(S, T)

    y_pred = predict(c, predict_year)
    relative_error = abs(y_pred - true_value) / true_value * 100

    if relative_error < best_error:
        best_error = relative_error
        best_degree_err = m

    print(f"Stopień {m}:")
    print(f"  Przewidziana populacja w 1990: {int(y_pred)}")
    print(f"  Błąd względny: {relative_error:.4f}%")

    k = m + 1
    n = len(x)

    rss = sum([(y[i] - predict(c, x[i]))**2 for i in range(n)])

    aic = 2*k + n * np.log(rss/n)
    aicc = aic + 2*k*(k+1) / (n-k-1)

    if aicc < best_aicc:
        best_aicc = aicc
        best_degree_aicc = m

    print(f"  AICc: {aicc:.4f}\n")

print(f"Najlepszy stopień według AICc: {best_degree_aicc} (AICc = {best_aicc:.4f})\n")
print(f"Najlepszy stopień według błędu: {best_degree_err} (Błąd względny = {best_error:.4f}%)\n")



# exercise 2

n_points = 100
k = np.arange(1, n_points + 1)
x_cheb = np.cos((2*k - 1) * np.pi / (2*n_points))
w_cheb = np.pi / n_points

def g(x):
    return np.sqrt(x + 1)

def T(k, x):
    if k==0:
        return np.ones_like(x)
    elif k==1:
        return x
    elif k==2:
        return 2 * x**2 - 1
    
coeffs = []

for i in range (3):
    integ = g(x_cheb) * T(i, x_cheb)
    if i==0:
        c = (1/np.pi) * np.sum(integ) * w_cheb
    else:
        c = (2/np.pi) * np.sum(integ) * w_cheb
    coeffs.append(c)

cheb_poly = Chebyshev(coeffs, domain = [-1, 1])

x_vals = np.linspace(0, 2, 400)
t_vals = x_vals - 1
approx_vals = cheb_poly(t_vals)
true_vals = np.sqrt(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, true_vals, label="sqrt(x)", linewidth=2)
plt.plot(x_vals, approx_vals, label="Aproksymacja Czebyszewa (stopień 2)", linestyle="--")
plt.title("Aproksymacja średniokwadratowa funkcji sqrt(x) wielomianem Czebyszewa")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.legend()
plt.show()

print("Współczynniki w bazie Czebyszewa:")
for i, c in enumerate(coeffs):
    print(f"c_{i} = {c:.6f}")