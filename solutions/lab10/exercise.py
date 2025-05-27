import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Parametry modelu
alpha1, alpha2 = 1, 0.5
beta1, beta2 = 0.1, 0.02
x0, y0 = 20, 20
T, h = 80, 0.1
N = int(T/h)

# Funkcje pochodnych
def dx(x, y): return x * (alpha1 - beta1 * y)
def dy(x, y): return y * (-alpha2 + beta2 * x)

# Invariant
def invariant(x, y): return beta2 * x + beta1 * y - alpha2 * np.log(x) - alpha1 * np.log(y)

# Metody numeryczne
def euler_explicit():
    x, y = np.zeros(N), np.zeros(N)
    x[0], y[0] = x0, y0
    for n in range(N-1):
        x[n+1] = x[n] + h * dx(x[n], y[n])
        y[n+1] = y[n] + h * dy(x[n], y[n])
    return x, y

def rk4():
    x, y = np.zeros(N), np.zeros(N)
    x[0], y[0] = x0, y0
    for n in range(N-1):
        k1x, k1y = dx(x[n], y[n]), dy(x[n], y[n])
        k2x = dx(x[n] + h*k1x/2, y[n] + h*k1y/2)
        k2y = dy(x[n] + h*k1x/2, y[n] + h*k1y/2)
        k3x = dx(x[n] + h*k2x/2, y[n] + h*k2y/2)
        k3y = dy(x[n] + h*k2x/2, y[n] + h*k2y/2)
        k4x = dx(x[n] + h*k3x, y[n] + h*k3y)
        k4y = dy(x[n] + h*k3x, y[n] + h*k3y)
        x[n+1] = x[n] + h * (k1x + 2*k2x + 2*k3x + k4x)/6
        y[n+1] = y[n] + h * (k1y + 2*k2y + 2*k3y + k4y)/6
    return x, y

def euler_semi_implicit():
    x, y = np.zeros(N), np.zeros(N)
    x[0], y[0] = x0, y0
    for n in range(N-1):
        y_next = y[n] / (1 - h * (-alpha2 + beta2 * x[n]))
        x[n+1] = x[n] + h * dx(x[n], y_next)
        y[n+1] = y_next
    return x, y

# Punkt stacjonarny
def stationary_points():
    xs = alpha2 / beta2
    ys = alpha1 / beta1
    return xs, ys

# Wykresy
def plot_all_methods():
    t = np.linspace(0, T, N)
    methods = {
        'Euler Jawny': euler_explicit(),
        'Półjawny Euler': euler_semi_implicit(),
        'RK4': rk4()
    }

    plt.figure(figsize=(12, 5))
    for label, (x, y) in methods.items():
        plt.plot(t, x, label=f'Ofiary ({label})')
        plt.plot(t, y, '--', label=f'Drapieżcy ({label})')
    plt.xlabel('Czas'); plt.ylabel('Liczebność'); plt.title('Populacje w czasie')
    plt.legend(); plt.grid(); plt.show()

    plt.figure()
    for label, (x, y) in methods.items():
        plt.plot(x, y, label=label)
    plt.xlabel('x'); plt.ylabel('y'); plt.title('Portret fazowy'); plt.legend(); plt.grid(); plt.show()

    # Niezmiennik
    plt.figure()
    for label, (x, y) in methods.items():
        H = invariant(x, y)
        plt.plot(t, H, label=label)
    plt.title("Niezmiennik H(x, y)")
    plt.xlabel("Czas"); plt.ylabel("H(x, y)"); plt.legend(); plt.grid(); plt.show()

# Estymacja parametrów
def simulate(theta, data_t):
    global alpha1, alpha2, beta1, beta2
    alpha1, alpha2, beta1, beta2 = theta
    x, y = np.zeros(len(data_t)), np.zeros(len(data_t))
    x[0], y[0] = data[0,1], data[0,0]
    for i in range(len(data_t)-1):
        k1x, k1y = dx(x[i], y[i]), dy(x[i], y[i])
        k2x = dx(x[i] + h*k1x/2, y[i] + h*k1y/2)
        k2y = dy(x[i] + h*k1x/2, y[i] + h*k1y/2)
        k3x = dx(x[i] + h*k2x/2, y[i] + h*k2y/2)
        k3y = dy(x[i] + h*k2x/2, y[i] + h*k2y/2)
        k4x = dx(x[i] + h*k3x, y[i] + h*k3y)
        k4y = dy(x[i] + h*k3x, y[i] + h*k3y)
        x[i+1] = x[i] + h * (k1x + 2*k2x + 2*k3x + k4x)/6
        y[i+1] = y[i] + h * (k1y + 2*k2y + 2*k3y + k4y)/6
    return x, y

def loss(theta):
    x_sim, y_sim = simulate(theta, np.arange(len(data)))
    return np.sum((data[:,0] - y_sim)**2 + (data[:,1] - x_sim)**2)

# Wczytaj dane i wykonaj optymalizację
data = np.loadtxt("LynxHare.txt")
res = minimize(loss, x0=[1, 0.5, 0.1, 0.02], method='Nelder-Mead')
print("Oszacowane parametry [alpha1, alpha2, beta1, beta2]:", res.x)

# Uruchom wykresy
plot_all_methods()
