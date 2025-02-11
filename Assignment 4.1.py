import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

file_path = r"C:\Users\shahi\Downloads\weight-height(1) (1).csv"
data = pd.read_csv(file_path)
print(data.head())
plt.figure(figsize=(10, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.5, color='blue')
plt.title('Scatter Plot of Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.show()

X = data['Height'].values.reshape(-1, 1)
y = data['Weight'].values
model = LinearRegression()
model.fit(X, y)

y_pred = model.predict(X)
plt.figure(figsize=(10, 6))
plt.scatter(data['Height'], data['Weight'], alpha=0.5, label='Data Points', color='blue')
plt.plot(data['Height'], y_pred, color='red', label='Regression Line')
plt.title('Linear Regression: Height vs Weight')
plt.xlabel('Height (inches)')
plt.ylabel('Weight (pounds)')
plt.legend()
plt.show()

rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")

