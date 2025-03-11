import numpy as np
import matplotlib.pyplot as plt

def f(x):
  return np.tan(x)

def df(x):
  return 1 + np.tan(x)**2

def df2(x):
  return 2*np.tan(x) * df(x)

def df_approx(x , h):
  return (f(x+h) - f(x)) / h

def epsilon(): #error for float64
  return 2.220446 * (10.0 ** -16)

x0 = 1
true_df = df(x0)
h_values = 10.0 ** -np.arange(0,17)
M = df2(x0)

def truncation_error(h):
  global M
  return M * h / 2.0

def rounding_error(h):
  return 2.0 * epsilon() / h

def computational_error(h):
  global x0
  return np.abs(df_approx(x0 , h) - df(x0))

h_min = 2 * np.sqrt(epsilon() / M)
print("h_min: ", h_min)
print("h_min abs error: ", np.abs(computational_error(h_min) / df(1)))

plt.figure(figsize=(8, 6))
plt.loglog(h_values, computational_error(h_values), label='Computational Error')
plt.loglog(h_values, truncation_error(h_values), label='Truncation Error')
plt.loglog(h_values, rounding_error(h_values), label='Rounding Error')
plt.xlabel('h')
plt.ylabel('Error')
plt.legend()
plt.title('Error analysis of the derivative approximation')
plt.grid(True, which='both', linestyle='--')
plt.show()