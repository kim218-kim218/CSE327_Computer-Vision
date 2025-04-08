# Assignment: Implementation of KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from collections import Counter

custom_flag = 1 #sklearn function:0, custom function:1

class CustomKNNClassifier:
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def DistanceMetric(self, x1, x2):
        if self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            raise ValueError(f"Cannot find distance matric")

    def predict(self, x):
        result = []
        for a in x:
            distances = [self.DistanceMetric(a, xt) for xt in self.x_train]
            indexPosition = np.argsort(distances)[:self.n_neighbors]
            nearestLabel = [self.y_train[i] for i in indexPosition]
            counter = Counter(nearestLabel)
            maxLabel = counter.most_common(1)[0][0]
            result.append(maxLabel)
        return np.array(result)

# Your Custom KNN Regressor with Distance Metric Selection
# class CustomKNNRegressor:
class CustomKNNRegressor:
    def __init__(self, n_neighbors, metric):
        self.n_neighbors = n_neighbors
        self.metric = metric

    def fit(self, x, y):
        self.x_train = x
        self.y_train = y

    def DistanceMetric(self, x1, x2):
        if self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        elif self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        else:
            raise ValueError(f"Cannot find distance matric")

    def predict(self, x):
        result = []
        for a in x:
            distances = [self.DistanceMetric(a, xt) for xt in self.x_train]
            indexPosition = np.argsort(distances)[:self.n_neighbors]
            nearestLabel = [self.y_train[i] for i in indexPosition]
            mean_value = np.mean(nearestLabel)
            result.append(mean_value)
        return np.array(result)

# 1D Classification
X_class = np.array([140, 145, 151, 159, 163, 169, 170, 174, 181, 182]).reshape(-1, 1)
y_class = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 = Short, 1 = Tall

if custom_flag == 0:
    knn_class_e = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_class_m = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
else:
    print("Custom")
    knn_class_e = CustomKNNClassifier(n_neighbors=5, metric='euclidean')
    knn_class_m = CustomKNNClassifier(n_neighbors=5, metric='manhattan')

knn_class_e.fit(X_class, y_class)
knn_class_m.fit(X_class, y_class)

new_height = np.array([[162]])
pred_class_e = knn_class_e.predict(new_height)
pred_class_m = knn_class_m.predict(new_height)
print(f"Predicted Class for 167 cm: euclidean= {'Tall' if pred_class_e[0] == 1 else 'Short'}")
print(f"Predicted Class for 167 cm: manhattan= {'Tall' if pred_class_m[0] == 1 else 'Short'}")

# 1D Regression
X_reg = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16]).reshape(-1, 1)
y_reg = np.array([11, 13, 12, 17, 16, 18, 16, 15, 15, 18, 22, 24, 20, 22])  # Temperature values

if custom_flag == 0:
    knn_temp_e = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
    knn_temp_m = KNeighborsRegressor(n_neighbors=5, metric='manhattan')
else:
    print("Custom")
    knn_temp_e = CustomKNNRegressor(n_neighbors=5, metric='euclidean')
    knn_temp_m = CustomKNNRegressor(n_neighbors=5, metric='manhattan')

knn_temp_e.fit(X_reg, y_reg)
knn_temp_m.fit(X_reg, y_reg)

new_hour = np.array([[13]])
pred_temp_e = knn_temp_e.predict(new_hour)
pred_temp_m = knn_temp_m.predict(new_hour)
print(f"Predicted Temperature at 11 AM: euclidean= {pred_temp_e[0]:.2f}°C, manhattan= {pred_temp_m[0]:.2f}°C")

# Visualization
plt.scatter(X_reg, y_reg, color="blue", label="Training Data")
plt.scatter(new_hour, pred_temp_e, color="red", label="Euclidean Prediction (11 AM)")
plt.scatter(new_hour, pred_temp_m, color="green", label="Manhattan Prediction (11 AM)")
plt.xlabel("Hour of the Day")
plt.ylabel("Temperature (°C)")
plt.title("1D KNN Regression")
plt.legend()
plt.show()