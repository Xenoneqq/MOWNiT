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
    matrix = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            matrix[i, j] = equation(X[i], j+1, n)
    return matrix

def interpolate(t, coeff, degree=0):
    if degree == len(coeff) - 2:
        return coeff[degree] + coeff[degree+1] * ((t-1940)/40)
    return coeff[degree] + ((t-1940)/40) * interpolate(t, coeff, degree+1)


# PDPKT A + B + C
x = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]
y = [76212168, 92228496, 106021537, 123202624, 132164569, 151325798, 179323175, 203302031, 226542199]

best_cond = math.inf
best_n = 0
for n in range(4):
    A = create_matrix(x, n)
    cond_number = np.linalg.cond(A)
    if cond_number < best_cond:
        best_cond = cond_number
        best_matrix = A
        best_n = n

print(f"Best condition number: {best_cond:.2e}")

a = np.linalg.solve(best_matrix, y)
#print(f"Współczynniki a: {a}")

x_vals = np.arange(1900, 1991, 1)
y_vals = [interpolate(t , a) for t in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Wielomian interpolacyjny", color='b')
plt.scatter(x, y, color='r', label="Węzły interpolacyjne")
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.title("Wielomian interpolacyjny")
plt.legend()
plt.grid()
plt.show()

# PDPKT D
year_1990_true = 248709873
year_predicted = interpolate(1990 , a)
print("Prawdziwa wartość dla 1990 roku: " , year_1990_true)
print("Przewidziana wartość dla 1990 roku: ", year_predicted)
relative_population_error = (abs(year_1990_true - year_predicted) / year_1990_true) * 100
print("Błąd względny ekstrapolacji dla roku 1990:", relative_population_error)

# PDPKT E
def lagrange_basis(x, i, t):
    result = 1.0
    for j in range(len(x)):
        if j != i:
            result *= (t - x[j]) / (x[i] - x[j])
    return result

def lagrange_interpolation(x, y, t):
    result = 0.0
    for i in range(len(x)):
        result += y[i] * lagrange_basis(x, i, t)
    return result

y_vals = np.array([lagrange_interpolation(x,y,t) for t in x_vals])

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Wielomian Lagrange’a', color='blue')
plt.scatter(x, y, color='red', label='Węzły interpolacji')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja populacji USA - Wielomian Lagrange’a')
plt.legend()
plt.grid()
plt.show()

# PDPKT F
def divided_differences(x, y):
    n = len(x)
    coef = np.array(y, dtype=float)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            coef[i] = (coef[i] - coef[i-1]) / (x[i] - x[i-j])
    return coef

def newton_interpolation(x, coef, t):
    n = len(coef)
    result = coef[-1]
    for i in range(n-2, -1, -1):
        result = result * (t - x[i]) + coef[i]
    return result

coef = divided_differences(x, y)

y_vals = np.array([newton_interpolation(x, coef, t) for t in x_vals])

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='Wielomian Newtona', color='blue')
plt.scatter(x, y, color='red', label='Węzły interpolacji')
plt.xlabel('Rok')
plt.ylabel('Populacja')
plt.title('Interpolacja populacji USA - Wielomian Newtona')
plt.legend()
plt.grid()
plt.show()

# PDPKT G
x = [1900, 1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980]
y = [76000000, 92000000, 106000000, 123000000, 132000000, 151000000, 179000000, 203000000, 227000000]


A_rounded = create_matrix(x, best_n)

a_rounded = np.linalg.solve(best_matrix, y)

y_vals = [interpolate(t , a_rounded) for t in x_vals]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label="Wielomian interpolacyjny", color='b')
plt.scatter(x, y, color='r', label="Węzły interpolacyjne")
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.title("Wielomian interpolacyjny")
plt.legend()
plt.grid()
plt.show()

print("Współczynniki przed zaokrągleniem")
print(a)
print("Współczynniki po zaokrągleniu")
print(a_rounded)

year_predicted_rounded = interpolate(1990 , a_rounded)
print("Prawdziwa wartość dla 1990 roku: " , year_1990_true)
print("Przewidziana wartość dla 1990 roku: ", year_predicted_rounded)
relative_population_error_rounded = (abs(year_1990_true - year_predicted_rounded) / year_1990_true) * 100
print("Błąd względny ekstrapolacji dla roku 1990:", relative_population_error_rounded)
