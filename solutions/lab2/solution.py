import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg

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

#Histogram_B()

# Pdpkt C
selected_features = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]
A_training = np.c_[np.ones(train_data.shape[0]), train_data.iloc[:,:2].values]
A_validate = np.c_[np.ones(validate_data.shape[0]), validate_data.iloc[:, 2:].values]

A_training_quad = np.c_[np.ones(train_data.shape[0]), train_data[selected_features].values, train_data[selected_features].pow(2).values]
A_validate_quad = np.c_[np.ones(validate_data.shape[0]), validate_data[selected_features].values, validate_data[selected_features].pow(2).values]
# print("Rozmiar macierzy A_train_lin:", A_training.shape)
# print("Rozmiar macierzy A_train_quad:", A_validate_quad.shape)

# pdpkt D
b_train = np.where(train_data["Malignant/Benign"] == "Malignant", 1, -1)
b_validate = np.where(validate_data["Malignant/Benign"] == "Malignant", 1, -1)
# print("Rozmiar wektora b_train:", b_train.shape)
# print("Rozmiar wektora b_validate:", b_validate.shape)

# pdpkt E

# tworzenie danych czysto numerycznych (bez naglowkow)
numeric_columns = train_data.select_dtypes(include=[np.number]).columns
numeric_columns = numeric_columns.difference(['patient ID', 'Malignant/Benign'])

A_training = np.c_[np.ones(train_data.shape[0]), train_data[numeric_columns].values] 
A_validate = np.c_[np.ones(validate_data.shape[0]), validate_data[numeric_columns].values]
A_training_quad = np.c_[np.ones(train_data.shape[0]), train_data[selected_features].values, train_data[selected_features].pow(2).values]
A_validate_quad = np.c_[np.ones(validate_data.shape[0]), validate_data[selected_features].values, validate_data[selected_features].pow(2).values]

# Pdpkt D
b_train = np.where(train_data["Malignant/Benign"] == "Malignant", 1, -1)
b_validate = np.where(validate_data["Malignant/Benign"] == "Malignant", 1, -1)

# Pdpkt E i F
A_linear = np.array(A_training)
b_training = np.array(b_train, dtype=int)
A_quad = np.array(A_training_quad)
b_training_quad = np.array(b_train, dtype=int)

lin_weight = np.linalg.solve(A_linear.T @ A_linear, A_linear.T @ b_training)
#print("Waga w reprezentacji liniowej: ", lin_weight)
lin_weight_quad = np.linalg.solve(A_quad.T @ A_quad, A_quad.T @ b_training_quad)
#print("Waga w reprezentacji kwadratowej: ", lin_weight_quad)
lin_weight_lstsq = scipy.linalg.lstsq(A_linear, b_training)[0]
#print("Waga w reprezentacji liniowej (lstsq):", lin_weight_lstsq)

# Pdpkt G
ATA_linear = A_linear.T @ A_linear
condition_num_linear = np.linalg.cond(ATA_linear)
print("wartość liniowej cond(ATA) : " , condition_num_linear)

ATA_quad = A_quad.T @ A_quad
condition_num_quad = np.linalg.cond(ATA_quad)
print("wartosc kwadratowe cond(ATA) : ", condition_num_quad)

# Pdpkt G


