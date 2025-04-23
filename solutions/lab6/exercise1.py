import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.integrate

def f(x):
    return 4 / (1 + x**2)

def midpoint_rectangular_formula(a, b, m):
    step = (b - a) / m
    result = 0
    for i in range(m):
        midpoint = a + (i + 0.5) * step
        result += f(midpoint)
    return result * step

def trapezoidal_formula(a, b, m):
    x = np.linspace(a, b, m)
    y = f(x)
    return scipy.integrate.trapezoid(y , x)

def simpsons_formula(a, b, m):
    # only works for odd number of points
    if m % 2 == 0:
        m += 1
    x = np.linspace(a, b, m)
    y = f(x)
    return scipy.integrate.simpson(y , x = x)

def relative_error(x):
    return abs(x - np.pi) / np.pi

# A

h_values = np.logspace(-2, -1, 20)
n_values = np.round(1 / h_values).astype(int)

mid_errors = [relative_error(midpoint_rectangular_formula(0, 1, n)) for n in n_values]
trap_errors = [relative_error(trapezoidal_formula(0, 1, n)) for n in n_values]
simp_errors = [relative_error(simpsons_formula(0, 1, n)) for n in n_values]

# zadanie 2

def f_trans(x):
    return f(x/2 + 1/2)

gauss_errors = []

for n in n_values:
    x, w = np.polynomial.legendre.leggauss(n)
    approx = np.sum(f_trans(x) * w) / 2
    gauss_errors.append(relative_error(approx))


plt.figure(figsize=(10, 6))
plt.plot(n_values + 1, mid_errors, label='Prostokąty (środek)', color='blue', marker='o')
plt.plot(n_values + 1, trap_errors, label='Trapezy', color='green', marker='x')
plt.plot(n_values + 1, simp_errors, label='Simpson', color='red', marker='s')
plt.plot(n_values + 1, gauss_errors, label='Gauss-Legendre', color='black',  marker='o', linestyle='-')
plt.yscale('log')
plt.xlabel('Liczba ewaluacji funkcji (n + 1)')
plt.ylabel('Błąd względny')
plt.title('Porównanie błędów względnych metod numerycznych całkowania')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# B

mid_min = np.min(mid_errors)
trap_min = np.min(trap_errors)
simp_min = np.min(simp_errors)

print("Najmniejszy błąd met. kwadratów : ", mid_min)
print("Najmniejszy błąd met. trapezów : ", trap_min)
print("Najmniejszy błąd met. simpsona : ", simp_min)

# Wynik porównać z lab 1

# C

def empirical_order(errors, h_values):
    p_values = []
    for i in range(len(errors) - 1):
        p = np.log(errors[i] / errors[i + 1]) / np.log(h_values[i] / h_values[i + 1])
        p_values.append(p)
    return np.mean(p_values)

mid_order = empirical_order(mid_errors, h_values)
trap_order = empirical_order(trap_errors, h_values)
simp_order = empirical_order(simp_errors, h_values)

print(f"Empiryczny rząd zbieżności metody prostokątów: {mid_order:.2f}")
print(f"Empiryczny rząd zbieżności metody trapezów: {trap_order:.2f}")
print(f"Empiryczny rząd zbieżności metody Simpsona: {simp_order:.2f}")