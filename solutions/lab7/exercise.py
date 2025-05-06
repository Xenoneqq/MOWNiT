import scipy.integrate as integrate
import matplotlib.pyplot as plt
import numpy as np

# Funkcja podcałkowa
def f(x):
    return 4 / (1 + x**2)

# Kwadratura trapezów
def trapezoid_rule(func, a, b, n):
    h = (b - a) / n
    sum_val = 0.5 * (func(a) + func(b))
    eval_count = 2
    for i in range(1, n):
        sum_val += func(a + i * h)
        eval_count += 1
    return sum_val * h, eval_count

# Kwadratura Gaussa-Kronroda
def adaptive_gauss_kronrod(func, a, b, tol):
    pack = integrate.quad(func, a, b, epsabs=tol, epsrel=tol, full_output=1)
    result = pack[0]
    neval = pack[2]['neval']
    return result, neval

# Zbiór tolerancji
tolerances = [10**(-i) for i in range(1, 15)]
trapezoid_errors = []
gauss_kronrod_errors = []
trapezoid_evals = []
gauss_kronrod_evals = []

# Testowanie na różnych tolerancjach
for tol in tolerances:
    # Kwadratura trapezów
    n = 1000  # Zwiększamy liczbę podziałów
    result_trap, evals_trap = trapezoid_rule(f, 0, 1, n)
    exact_value = np.pi
    trapezoid_errors.append(abs(result_trap - exact_value))
    trapezoid_evals.append(evals_trap)

    # Kwadratura Gaussa-Kronroda
    result_gauss, evals_gauss = adaptive_gauss_kronrod(f, 0, 1, tol)
    gauss_kronrod_errors.append(abs(result_gauss - exact_value))
    gauss_kronrod_evals.append(evals_gauss)

# Sprawdzenie danych
print("Trapezoid errors:", trapezoid_errors)
print("Gauss-Kronrod errors:", gauss_kronrod_errors)

plt.figure(figsize=(10, 6))
plt.loglog(trapezoid_evals, trapezoid_errors, label="Kwadratura Trapezów", marker='o')
plt.loglog(gauss_kronrod_evals, gauss_kronrod_errors, label="Kwadratura Gaussa-Kronroda", marker='x')
plt.xlabel("Liczba ewaluacji funkcji")
plt.ylabel("Błąd względny")
plt.legend()
plt.title("Błąd względem liczby ewaluacji funkcji")
plt.grid(True)
plt.show()
