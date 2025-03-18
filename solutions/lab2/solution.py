import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from sklearn.metrics import confusion_matrix

# Pdpkt A
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

columns = pd.read_csv(os.path.join(dataset_dir, "breast-cancer.labels"), header=None).squeeze().tolist()
train_data = pd.read_csv(os.path.join(dataset_dir, "breast-cancer-train.dat"), names=columns)
validate_data = pd.read_csv(os.path.join(dataset_dir, "breast-cancer-validate.dat"), names=columns)

# Pdpkt B
malignant = train_data[train_data["Malignant/Benign"] == "M"]
benign = train_data[train_data["Malignant/Benign"] == "B"]
feature = "radius (mean)"
plt.figure(figsize=(8, 5))
plt.hist(malignant[feature], bins=20, alpha=0.7, label="Złośliwy", color="red")
plt.hist(benign[feature], bins=20, alpha=0.7, label="Łagodny", color="blue")
plt.xlabel("Średni promień")
plt.ylabel("Liczba pacjentów")
plt.title("Histogram zależności średniego promienia, a liczny wystąpień")
plt.legend()
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.yticks(np.arange(0, 20, 2))
# plt.show()

# Pdpkt C
selected_features = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]
A_training_lin = np.c_[np.ones(train_data.shape[0]), train_data.iloc[:, 2:].values]
A_validate_lin = np.c_[np.ones(validate_data.shape[0]), validate_data.iloc[:, 2:].values]

A_training_quad = np.c_[np.ones(train_data.shape[0]), train_data[selected_features].values, train_data[selected_features].pow(2).values]
A_validate_quad = np.c_[np.ones(validate_data.shape[0]), validate_data[selected_features].values, validate_data[selected_features].pow(2).values]

# print("Rozmiar macierzy A_train_lin: ", A_training_lin.shape)
# print("Rozmiar macierzy A_train_quad: ", A_training_quad.shape)
# print("Rozmiar macierzy A_validate_lin: ", A_validate_lin.shape)
# print("Rozmiar macierzy A_validate_quad: ", A_validate_quad.shape)


# pdpkt D
b_training = np.where(train_data["Malignant/Benign"] == "M", 1, -1)
b_validate = np.where(validate_data["Malignant/Benign"] == "M", 1, -1)

# print("Rozmiar wektora b_train:", b_train.shape)
# print("Rozmiar wektora b_validate:", b_validate.shape)

# pdpkt E
# wektor_wag = (ATA)-1 AT b

lin_weight = np.linalg.solve(A_training_lin.T @ A_training_lin, A_training_lin.T @ b_training)
quad_weight = np.linalg.solve(A_training_quad.T @ A_training_quad, A_training_quad.T @ b_training)

# print("Waga w reprezentacji liniowej: ")
# for i in range(1, len(columns)-1):
#     if i == 1: print("{:<30} {:>10.6f}".format("bias ", lin_weight[i]))
#     else: print("{:<30} {:>10.6f}".format(columns[i], lin_weight[i]))
# print("Waga w reprezentacji kwadratowej: ")
# for i in range(2*len(selected_features) + 1):
#     if i == 0: print("{:<30} {:>10.6f}".format("bias ", quad_weight[i]))
#     elif i <= len(selected_features): print("{:<30} {:>10.6f}".format(selected_features[i-1], quad_weight[i]))
#     else: print("{:<30} {:>10.6f}".format(selected_features[i-len(selected_features)-1] + " kwadrat", quad_weight[i]))


# pdpkt F
lin_weight_lstsq = scipy.linalg.lstsq(A_training_lin, b_training)[0]
lambda_ = 0.01
I = np.eye(A_training_lin.shape[1])
I[0,0] = 0
w_ridge = np.linalg.solve(A_training_lin.T @ A_training_lin + lambda_ * I, A_training_lin.T @ b_training)

# print("Waga w reprezentacji liniowej (lstsq): ")
# for i in range(1, len(columns)-1):
#     if i == 1: print("{:<30} {:>10.6f}".format("bias ", lin_weight_lstsq[i]))
#     else: print("{:<30} {:>10.6f}".format(columns[i], lin_weight_lstsq[i]))
# 
# print("Waga w reprezentacji liniowej z regularyzacją λ=0.01: ")
# for i in range(1, len(columns)-1):
#     if i == 1: print("{:<30} {:>10.6f}".format("bias ", w_ridge[i]))
#     else: print("{:<30} {:>10.6f}".format(columns[i], w_ridge[i]))

# Pdpkt G
ATA_linear = A_training_lin.T @ A_training_lin
condition_num_linear = np.linalg.cond(ATA_linear)

ATA_quad = A_training_quad.T @ A_training_quad
condition_num_quad = np.linalg.cond(ATA_quad)

# print("Wartość cond(ATA) dla liniowej metody najmniejszych kwadratów: " , condition_num_linear)
# print("Wartość cond(ATA) dla kwadratowej metody najmniejszych kwadratów: ", condition_num_quad)

# Pdpkt H
p_lin = A_validate_lin @ lin_weight
p_quad = A_validate_quad @ quad_weight

predictions_lin = np.where(p_lin > 0, 1, -1)
conf_matric_lin = confusion_matrix(b_validate, predictions_lin)
TP = conf_matric_lin[1, 1] # złośliwy
TN = conf_matric_lin[0, 0] # łagodny
FP = conf_matric_lin[0, 1] # łagodny jako złośliwy
FN = conf_matric_lin[1, 0] # złośliwy jako łagodny
lin_acc = (TP + TN) / (TP + TN + FP + FN)

predictions_quad = np.where(p_quad > 0, 1, -1)
conf_matric_quad = confusion_matrix(b_validate, predictions_quad)
TP = conf_matric_quad[1, 1] # złośliwy
TN = conf_matric_quad[0, 0] # łagodny
FP = conf_matric_quad[0, 1] # łagodny jako złośliwy
FN = conf_matric_quad[1, 0] # złośliwy jako łagodny
quad_acc = (TP + TN) / (TP + TN + FP + FN)

# print("Macierz pomyłek dla metody liniowej\n", conf_matric_lin)
# print("Dokładność: ", round(lin_acc, 2), "\n")
# print("Macierz pomyłek dla metody kwadratowej\n", conf_matric_quad)
# print("Dokładność: ", round(quad_acc, 2))
