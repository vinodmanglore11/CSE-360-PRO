import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("crop_yield_dataset (1).csv")
print(df)
X = df[['rainfall', 'temperature', 'soil_moisture',
        'fertilizer', 'area']]
y = df['yield']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)

print("\nEnter details for crop yield prediction")

rainfall = float(input("Enter rainfall (mm): "))
temperature = float(input("Enter temperature (°C): "))
soil_moisture = float(input("Enter soil moisture (%): "))
fertilizer = float(input("Enter fertilizer (kg): "))
area = float(input("Enter area (hectare): "))

new_data = [[
    rainfall,
    temperature,
    soil_moisture,
    fertilizer,
    area
]]

prediction = model.predict(new_data)

print("\nPredicted Crop Yield:",
    round(prediction[0], 2),
    "tons/hectare")
