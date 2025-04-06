# Assignment: Implementation of KNN

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

custom_flag = 1 #sklearn function:0, custom function:1

# Your Custom KNN Classifier with Distance Metric Selection
# class CustomKNNClassifier:
class CustomKNNClassifier:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.k = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._compute_distance(x, train_x) for train_x in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(most_common)
        return np.array(predictions)

# Your Custom KNN Regressor with Distance Metric Selection
# class CustomKNNRegressor:
class CustomKNNRegressor:
    def __init__(self, n_neighbors=3, metric='euclidean'):
        self.k = n_neighbors
        self.metric = metric

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _compute_distance(self, x1, x2):
        if self.metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2))
        elif self.metric == 'manhattan':
            return np.sum(np.abs(x1 - x2))
        else:
            raise ValueError(f"Unsupported distance metric: {self.metric}")

    def predict(self, X):
        predictions = []
        for x in X:
            distances = [self._compute_distance(x, train_x) for train_x in self.X_train]
            k_indices = np.argsort(distances)[:self.k]
            k_nearest_values = [self.y_train[i] for i in k_indices]
            mean_value = np.mean(k_nearest_values)
            predictions.append(mean_value)
        return np.array(predictions)

# 1D Classification
X_class = np.array([140, 145, 151, 159, 163, 169, 170, 174, 181, 182]).reshape(-1, 1)
y_class = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])  # 0 = Short, 1 = Tall

if custom_flag == 0:
    knn_class_e = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    knn_class_m = KNeighborsClassifier(n_neighbors=5, metric='manhattan')
else:
    print("my")
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
    print("my")
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