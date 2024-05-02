# STEP 15
# Deployment of the best model in production

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

file_path = 'C:\\Users\\kevin\\Downloads\\Laptop_price.csv'
df = pd.read_csv(file_path)

# Define features and target variable
X = df[['Processor_Speed', 'RAM_Size', 'Storage_Capacity']]
y = df['Price']

# Train the final model
final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])

# Determined in step 14 that Linear Regression is the best modle. 

final_model.fit(X, y)

# Save the final model
joblib.dump(final_model, 'final_model.pkl')

# Load the trained model
final_model = joblib.load('final_model.pkl')

# Define color dictionary
color = {
    "RED": "\033[91m",
    "GREEN": "\033[92m",
    "BOLD": "\033[1m",
    "END": "\033[0m"
}

# Get user input
def get_user_data():
    user_data = {}
    user_data['Processor_Speed'] = float(input("Enter Processor Speed: "))
    user_data['RAM_Size'] = int(input("Enter RAM Size: "))
    user_data['Storage_Capacity'] = int(input("Enter Storage Capacity: "))
    return user_data

# ONLY asks the user for these 3 inputs because we determined only these matter the most in STEP 10. 

# Predict with the model
def predict_price(model, user_data):
    X_user = pd.DataFrame([user_data])
    predicted_price = model.predict(X_user)[0]
    return predicted_price

# Get user input
user_data = get_user_data()

# Predict with the model
predicted_price = predict_price(final_model, user_data)

# Display the prediction
print(f"Predicted price: {predicted_price}")

# Display the predicted price
# Display the predicted price
print(f"Predicted price before purchasing:{color['BOLD']}{color['GREEN']}${predicted_price}{color['END']}")


