import numpy as np
import matplotlib.pyplot as plt

n_values = [500, 1000, 2000, 5000, 10000, 15000, 20000, 50000, 100000]

for n in n_values:
    dice1 = np.random.randint(1, 7, n)
    dice2 = np.random.randint(1, 7, n)
    s = dice1 + dice2
    h, h2 = np.histogram(s, bins=range(2, 14))  # Bin edges from 2 to 12
    plt.figure(figsize=(10, 5))
    plt.bar(h2[:-1], h / n, width=0.8, color='skyblue', edgecolor='black')  # Normalized frequency
    plt.title(f'Histogram of Dice Sums (n={n})')
    plt.xlabel('Sum of Two Dice')
    plt.ylabel('Relative Frequency')
    plt.xticks(range(2, 13))
    plt.show()
