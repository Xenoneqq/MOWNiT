import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Pdpkt A
columns = pd.read_csv("dataset/breast-cancer.labels", header=None).squeeze().tolist()
train_data = pd.read_csv("dataset/breast-cancer-train.dat", names=columns)
validate_data = pd.read_csv("dataset/breast-cancer-validate.dat", names=columns)

# Pdpkt B
def Histogram_B():
  malignant = train_data[train_data["Malignant/Benign"] == "M"]
  benign = train_data[train_data["Malignant/Benign"] == "B"]
  feature = "radius (mean)"

  plt.figure(figsize=(8, 5))
  plt.hist(malignant[feature], bins=20, alpha=0.7, label="Malignant", color="red")
  plt.hist(benign[feature], bins=20, alpha=0.7, label="Benign", color="blue")

  plt.xlabel(feature)
  plt.ylabel("Liczba pacjentów")
  plt.title(f"Histogram wartości cechy {feature}")
  plt.legend()
  plt.grid(color='gray', linestyle='--', linewidth=0.5)
  plt.yticks(np.arange(0, 20, 2))
  plt.show()

Histogram_B()
