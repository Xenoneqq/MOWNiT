import numpy as np
import math
import matplotlib.pyplot as plt

def equation(t, j, n):
    match n:
        case 0:
            return t**(j-1)
        case 1:
            return (t - 1900)**(j-1)
        case 2:
            return (t-1940)**(j-1)
        case 3:
            return ((t-1940)/40)**(j-1)
    return 1

def create_matrix(X, n):
    size = len(X)
    A = np.zeros((size, size))
    for i in range(size):
        for j in range(1, size + 1):
            A[i, j - 1] = equation(X[i], j, n)
    return A

def horner(coeffs, t):
    result = coeffs[-1]
    for i in range(len(coeffs) - 2, -1, -1):
        result = result * t + coeffs[i]
    return result

x = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]
y = [76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199]

best_cond = math.inf
for n in range(4):
    A = create_matrix(x, n)
    cond_number = np.linalg.cond(A)
    if cond_number < best_cond:
        best_cond = cond_number
        best_matrix = A

print(f"Best condition number: {best_cond:.2e}")

a = np.linalg.solve(best_matrix, y)

x_vals = np.arange(1900, 1991, 1)
y_vals = [horner(a, t) for t in x_vals]

print(f"Współczynniki a: {a}")

plt.plot(x_vals, y_vals, label="Wielomian interpolacyjny", color='b')
plt.scatter(x, y, color='r', label="Węzły interpolacyjne")
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.title("Wielomian interpolacyjny")
plt.legend()
plt.grid(True)
plt.show()
