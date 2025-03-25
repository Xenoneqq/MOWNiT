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
    matrix = np.zeros(shape=(9, 9))
    for i in range(9):
        for j in range(9):
            matrix[i][j] = equation(X[i], j+1, n)
    matrix = np.matrix(matrix)
    return matrix

def interpolate(t, coeff, degree=0):
    if degree == len(coeff) - 2:
        return coeff[degree] + coeff[degree+1] * ((t-1940)/40)
    return coeff[degree] + ((t-1940)/40) * interpolate(t, coeff, degree+1)


# PDPKT A + B + C

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
#print(f"Współczynniki a: {a}")

x_vals = np.arange(1900, 1991, 1)
y_vals = [interpolate(t , a) for t in x_vals]

# rysowanie wykresu na ekran (pdpkt c)
plt.plot(x_vals, y_vals, label="Wielomian interpolacyjny", color='b')
plt.scatter(x, y, color='r', label="Węzły interpolacyjne")
plt.xlabel("Rok")
plt.ylabel("Populacja")
plt.title("Wielomian interpolacyjny")
plt.legend()
plt.grid(True)
plt.show()

# PDPKT D

year_1990_true = 248709873
year_predicted = interpolate(1990 , a)
print("True 1990 value : " , year_1990_true)
print("Predicted 1990 value : ", year_predicted)
relative_population_error = (abs(year_1990_true - year_predicted) / year_1990_true) * 100
print("Relative error of extrapolated population:", relative_population_error)