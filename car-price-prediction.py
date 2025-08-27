                         # car-price-prediction #


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Create synthetic dataset
data = {
    'brand': ['Toyota','BMW','Honda','Ford','Audi','Toyota','BMW','Honda'],
    'horsepower': [120, 200, 110, 150, 180, 140, 250, 130],
    'mileage': [50000, 30000, 60000, 45000, 25000, 70000, 20000, 65000],
    'year': [2015, 2018, 2014, 2016, 2019, 2013, 2020, 2014],
    'price': [8000, 25000, 7500, 12000, 28000, 6000, 32000, 7200]
}

df = pd.DataFrame(data)
print(df)

# Add new column: age of the car
df['age'] = 2025 - df['year']

# One-hot encode categorical column 'brand'
df = pd.get_dummies(df, columns=['brand'], drop_first=True)

print(df.head())

X = df.drop(columns=['price'])
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R² Score:", r2)

plt.scatter(y_test, y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Car Prices")
plt.show()
