import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import time

# _______________ CZYTANIE I SELEKCJA DANYCH _______________
dataset_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")

columns = pd.read_csv(os.path.join(dataset_dir, "breast-cancer.labels"), header=None).squeeze().tolist()
train_data = pd.read_csv(os.path.join(dataset_dir, "breast-cancer-train.dat"), names=columns)
validate_data = pd.read_csv(os.path.join(dataset_dir, "breast-cancer-validate.dat"), names=columns)

selected_features = ["radius (mean)", "perimeter (mean)", "area (mean)", "symmetry (mean)"]
A_training_lin = np.c_[np.ones(train_data.shape[0]), train_data.iloc[:, 2:].values]
A_validate_lin = np.c_[np.ones(validate_data.shape[0]), validate_data.iloc[:, 2:].values]

interaction_terms = [train_data[selected_features[i]] * train_data[selected_features[j]] for i in range(len(selected_features)-1) for j in range(i, len(selected_features))]
interaction_terms_validate = [validate_data[selected_features[i]] * validate_data[selected_features[j]] for i in range(len(selected_features)-1) for j in range(i, len(selected_features))]

A_training_quad = np.c_[np.ones(train_data.shape[0]), train_data[selected_features].values, 
                         np.column_stack(interaction_terms)]
A_validate_quad = np.c_[np.ones(validate_data.shape[0]), validate_data[selected_features].values, 
                         np.column_stack(interaction_terms_validate)]

b_training = np.where(train_data["Malignant/Benign"] == "M", 1, -1)
b_validate = np.where(validate_data["Malignant/Benign"] == "M", 1, -1)

# _______________ GRADIENT DESCENT _______________
def gradient_descent(A, b, max_iter=100000, tol=1e-10):
    ATA = A.T @ A
    ATb = A.T @ b
    eigvals = np.linalg.eigvalsh(ATA)
    alpha = 2 / (eigvals.max() + eigvals.min())
    beta = 0.99
    lbda = 0.99

    w = np.zeros(A.shape[1])
    losses = []
    start = time.time()

    v = 0

    for i in range(max_iter):
        grad = ATA @ w - ATb
        v_next = beta*v + (1-beta)*grad #with momentum
        # v_next = beta*v + (1-beta)* (ATA @ (w - alpha*beta*v) - ATb) #Nestrov
        # w_new = w - alpha * grad #fixed alpha
        # w_new = alpha * (lbda ** i) * grad #decaying alpha
        # w_new = w - alpha / (1 + lbda * i) * grad #inverse decaying alpha
        w_new = w - alpha * v_next #with momentum/Nestrov
        loss = 0.5 * np.linalg.norm(A @ w_new - b) ** 2
        losses.append(loss)

        if np.linalg.norm(w_new - w) < tol:
            break

        w = w_new
        v = v_next

    end = time.time()
    return w, losses, end - start

w_gd_lin, losses_lin, time_lin = gradient_descent(A_training_lin, b_training)
w_gd_quad, losses_quad, time_quad = gradient_descent(A_training_quad, b_training)

# _______________ DOKŁADNOŚĆ _______________
predictions_gd_lin = np.where(A_validate_lin @ w_gd_lin > 0, 1, -1)
predictions_gd_quad = np.where(A_validate_quad @ w_gd_quad > 0, 1, -1)

cm_gd_lin = confusion_matrix(b_validate, predictions_gd_lin)
cm_gd_quad = confusion_matrix(b_validate, predictions_gd_quad)

acc_gd_lin = (cm_gd_lin[0, 0] + cm_gd_lin[1, 1]) / np.sum(cm_gd_lin)
acc_gd_quad = (cm_gd_quad[0, 0] + cm_gd_quad[1, 1]) / np.sum(cm_gd_quad)

print("Gradient Descent - Liniowy")
print("Macierz pomyłek:\n", cm_gd_lin)
print("Dokładność:", round(acc_gd_lin, 2), "Czas:", round(time_lin, 6), "s\n")

print("Gradient Descent - Kwadratowy")
print("Macierz pomyłek:\n", cm_gd_quad)
print("Dokładność:", round(acc_gd_quad, 2), "Czas:", round(time_quad, 6), "s\n")

# plt.plot(losses_lin, label="Liniowy")
# plt.plot(losses_quad, label="Kwadratowy")
# plt.yscale("log")
# plt.xlabel("Iteracja")
# plt.ylabel("Funkcja kosztu")
# plt.title("Zbieżność gradient descent")
# plt.legend()
# plt.grid(True)
# plt.show()

# _______________ RSS _______________ (skopiowane z lab2)
start = time.time()
lin_weight = np.linalg.solve(A_training_lin.T @ A_training_lin, A_training_lin.T @ b_training)
lin_time = time.time() - start
start = time.time()
quad_weight = np.linalg.solve(A_training_quad.T @ A_training_quad, A_training_quad.T @ b_training)
quad_time = time.time() - start

lambda_ = 0.01
I = np.eye(A_training_lin.shape[1])
I[0,0] = 0
start = time.time()
w_ridge = np.linalg.solve(A_training_lin.T @ A_training_lin + lambda_ * I, A_training_lin.T @ b_training)
ridge_time = time.time() - start

p_lin = A_validate_lin @ lin_weight
p_quad = A_validate_quad @ quad_weight
p_ridge = A_validate_lin @ w_ridge

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

predictions_ridge = np.where(p_ridge > 0, 1, -1)
conf_matric_ridge = confusion_matrix(b_validate, predictions_ridge)
TP = conf_matric_ridge[1, 1] # złośliwy
TN = conf_matric_ridge[0, 0] # łagodny
FP = conf_matric_ridge[0, 1] # łagodny jako złośliwy
FN = conf_matric_ridge[1, 0] # złośliwy jako łagodny
ridge_acc = (TP + TN) / (TP + TN + FP + FN)

print("Macierz pomyłek dla metody liniowej\n", conf_matric_lin)
print("Dokładność: ", round(lin_acc, 2), "Czas:", round(lin_time, 6), "s\n")
print("Macierz pomyłek dla metody kwadratowej\n", conf_matric_quad)
print("Dokładność: ", round(quad_acc, 2), "Czas:", round(quad_time, 6), "s\n")
print("Macierz pomyłek dla metody liniowej z regularyzacją\n", conf_matric_ridge)
print("Dokładność: ", round(ridge_acc, 2), "Czas:", round(ridge_time, 6), "s\n")