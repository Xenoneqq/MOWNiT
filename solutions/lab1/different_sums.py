import numpy as np
import matplotlib.pyplot as plt
import math

def generate_numbers(n):
    return np.random.uniform(0, 1, n).astype(np.float32)

def true_sum_value(numbers):
    return math.fsum(numbers)

def sum_double_precision(numbers):
    acc = np.float64(0.0)
    for num in numbers:
        acc += np.float64(num)
    return acc

def sum_single_precision(numbers):
    acc = np.float32(0.0)
    for num in numbers:
        acc += np.float32(num)
    return acc

def sum_kahan_alg(numbers):
    acc = np.float32(0.0)
    err = np.float32(0.0)
    for num in numbers:
        y = num - err
        temp = acc + y
        err = (temp - acc) - y
        acc = temp
    return acc

def sum_rising(numbers):
    return sum_single_precision(np.sort(numbers))

def sum_falling(numbers):
    return sum_single_precision(np.flip(np.sort(numbers)))

n_array = np.array([10 ** k for k in range (4,9)])
methods = ['a', 'b', 'c', 'd', 'e']
errors = {method: [] for method in methods}

for n in n_array:
    numbers = generate_numbers(n)
    true_sum = true_sum_value(numbers)

    errors['a'].append(abs(sum_double_precision(numbers) - true_sum) / true_sum)
    errors['b'].append(abs(sum_single_precision(numbers) - true_sum) / true_sum)
    errors['c'].append(abs(sum_kahan_alg(numbers) - true_sum) / true_sum)
    errors['d'].append(abs(sum_rising(numbers) - true_sum) / true_sum)
    errors['e'].append(abs(sum_falling(numbers) - true_sum) / true_sum)

plt.figure(figsize = (10, 6))
for method in methods:
    plt.loglog(n_array, errors[method], 'o-', label = f'Metoda {method}')
plt.xlabel('n (log)')
plt.ylabel('Błąd względny (log)')
plt.title('Porównanie błędów względnych metod sumowania')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.xticks(n_array, [f'10^{k}' for k in range(4,9)])

all_values = np.concatenate([errors[method] for method in methods])
ymin, ymax = np.min(all_values), np.max(all_values)
plt.ylim(ymin * 0.5, ymax * 2) 

plt.show()
