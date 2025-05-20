from scipy.optimize import root
import numpy as np

def f_vec(v):
    x1, x2 = v
    f1 = x1**2 + x2**2 - 1
    f2 = x1**2 - x2
    return [f1, f2]

def x_n_plus_1(vn):
    x1, x2 = vn
    jakobian_reversed = np.linalg.inv([[2*x1, 2*x2], [2*x1, -1]])

    return vn - jakobian_reversed @ f_vec(vn)

def x_newtona(x0):
    x1 = x_n_plus_1(x0)
    while(np.linalg.norm(x1-x0) > 0.0001):
        x0, x1 = x1, x_n_plus_1(x1)
    return x1

x0 = [0.5, 0.2]

approx = x_newtona(x0)
print("Obliczone x^Newtona: ", approx)

true_x1 = np.sqrt(np.sqrt(5)/2 - 0.5)
true_x2 = np.sqrt(5)/2 - 0.5

relative_error = np.linalg.norm(approx - [true_x1, true_x2]) /\
np.linalg.norm([true_x1, true_x2])

print("Błąd względny: ",relative_error)