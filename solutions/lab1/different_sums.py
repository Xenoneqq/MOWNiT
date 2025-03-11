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

# Prevents value of zero from happening
def safe_sum(value, epsilon=1e-20):
    return value if abs(value) > epsilon else epsilon

n_array = np.array([10 ** k for k in range(4, 9)])
max_n = n_array[-1]

numbers = generate_numbers(max_n)

methods = ['a', 'b', 'c', 'd', 'e']
errors = {method: [] for method in methods}

for n in n_array:
    subset = numbers[:n]
    true_sum = true_sum_value(subset)
    print("Generated numbers")

    print("Generating task a...")
    errors['a'].append(abs(safe_sum(sum_double_precision(subset)) - true_sum) / true_sum)
    print("Done...")

    print("Generating task b...")
    errors['b'].append(abs(safe_sum(sum_single_precision(subset)) - true_sum) / true_sum)
    print("Done...")

    print("Generating task c...")
    kahan_error = abs(safe_sum(sum_kahan_alg(subset)) - true_sum) / true_sum
    errors['c'].append(kahan_error if kahan_error > 0 else 1e-20)
    print("Done...")

    print("Generating task d...")
    errors['d'].append(abs(safe_sum(sum_rising(subset)) - true_sum) / true_sum)
    print("Done...")

    print("Generating task e...")
    errors['e'].append(abs(safe_sum(sum_falling(subset)) - true_sum) / true_sum)
    print("Done...")

    print("Done the subset of: ", n)

plt.figure(figsize=(10, 6))
for method in methods:
    plt.loglog(n_array, errors[method], 'o-', label=f'Metoda {method}')
plt.xlabel('n (log)')
plt.ylabel('Błąd względny (log)')
plt.title('Porównanie błędów względnych metod sumowania')
plt.legend()
plt.grid(True, which='both', linestyle='--')
plt.xticks(n_array, [f'10^{k}' for k in range(4, 9)])

all_values = np.concatenate([errors[method] for method in methods])
ymin, ymax = np.min(all_values), np.max(all_values)
plt.ylim(ymin * 0.5, ymax * 2)

plt.show()
