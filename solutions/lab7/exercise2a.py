import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

def f(x):
    return np.sqrt(x) * np.log(x)

true_value = -4/9

# Z poprzedniego zadania

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
    return integrate.trapezoid(y , x)

def simpsons_formula(a, b, m):
    # only works for odd number of points
    if m % 2 == 0:
        m += 1
    x = np.linspace(a, b, m)
    y = f(x)
    return integrate.simpson(y , x = x)

def relative_error(x):
    return abs((x - true_value) / true_value)

h_values = np.logspace(-1, -3, 20)
n_values = np.round(1 / h_values).astype(int)

mid_errors = [relative_error(midpoint_rectangular_formula(1e-16, 1, n)) for n in n_values]
trap_errors = [relative_error(trapezoidal_formula(1e-16, 1, n)) for n in n_values]
simp_errors = [relative_error(simpsons_formula(1e-16, 1, n)) for n in n_values]

def f_trans(x):
    return f(x/2 + 1/2)

gauss_errors = []

for n in n_values:
    x, w = np.polynomial.legendre.leggauss(n)
    approx = np.sum(f_trans(x) * w) / 2
    gauss_errors.append(relative_error(approx))


# Zadanie z tego tygodnia

# a) Kwadratura adapcyjna trapezów

def adaptive_trapezoid(f, a, b, tolerance, max_depth = 20):
    count = [0]
    
    def recursive(f, a, b, fa, fb, prev, tolerance, depth):
        mid = (a+b) / 2
        fm = f(mid)
        count[0] += 1
        left = (fa + fm) * (mid - a) / 2
        right = (fm + fb) * (b - mid) / 2
        if abs(prev - (left + right)) < tolerance or depth >= max_depth:
            return left + right
        else:
            return recursive(f, a, mid, fa, fm, left, tolerance / 2, depth + 1) +\
                   recursive(f, mid, b, fm, fb, right, tolerance / 2, depth + 1)

    fa, fb = f(a), f(b)
    count[0] += 2
    initial = (fa + fb) * (b - a) / 2
    result = recursive(f, a, b, fa, fb, initial, tolerance, 0)
    return result, count[0]

# b) Kwadratura adapcyjna Gaussa-Kronroda

def gauss_kronrod(f, a, b, tolerance):
    result, error, info = integrate.quad_vec(f, a, b, epsabs=tolerance, full_output=1)
    return result, info.neval

tolerances = np.logspace(-2, -14, 13)

trap_errs = []
trap_evals = []
gk_errs = []
gk_evals = []

for tol in tolerances:
    result, evals = adaptive_trapezoid(f, 1e-16, 1, tol)
    trap_errs.append(relative_error(result))
    trap_evals.append(evals)

    result, evals = gauss_kronrod(f, 0, 1, tol)
    gk_errs.append(relative_error(result))
    gk_evals.append(evals)


plt.figure(figsize=(10, 6))
plt.plot(n_values + 1, mid_errors, label='Prostokąty (środek)', color='blue', marker='o')
plt.plot(n_values + 1, trap_errors, label='Trapezy', color='green', marker='x')
plt.plot(n_values + 1, simp_errors, label='Simpson', color='red', marker='s')
plt.plot(n_values + 1, gauss_errors, label='Gauss-Legendre', color='black',  marker='o', linestyle='-')

plt.plot(trap_evals, trap_errs, label='Adapcyjne Trapezy', color='purple', marker='D')

plt.plot(gk_evals, gk_errs, label='Gauss-Kronrod', color='orange', marker='^')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Liczba ewaluacji funkcji (n + 1)')
plt.ylabel('Błąd względny')
plt.title('Porównanie błędów względnych metod numerycznych całkowania')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()