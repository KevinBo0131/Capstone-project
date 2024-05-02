import tkinter as tk
from tkinter import messagebox
import joblib
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the trained model (Refer back to STEP 15)
final_model = joblib.load('final_model.pkl')

# Function to predict price
def predict_price():
    try:
        # Get user input
        user_data = {
            'Processor_Speed': float(processor_speed_entry.get()),
            'RAM_Size': int(ram_size_entry.get()),
            'Storage_Capacity': int(storage_capacity_entry.get())
        }

        # Predict with the model
        predicted_price = final_model.predict(pd.DataFrame([user_data]))[0]

        # Display the prediction
        predicted_price_label.config(text=f"Predicted price: ${predicted_price:.2f}")

    except Exception as e:
        messagebox.showerror("Error", "Invalid input!")

# Function to reset entries
def reset_entries():
    processor_speed_entry.delete(0, tk.END)
    ram_size_entry.delete(0, tk.END)
    storage_capacity_entry.delete(0, tk.END)
    predicted_price_label.config(text="")

# Create main window
root = tk.Tk()
root.title("Laptop Price Prediction")

# Labels and Entries
labels = ["Processor Speed:", "RAM Size:", "Storage Capacity:"]
entries = []

for i, label_text in enumerate(labels):
    label = tk.Label(root, text=label_text)
    label.grid(row=i, column=0, padx=10, pady=5)
    entry = tk.Entry(root)
    entry.grid(row=i, column=1, padx=10, pady=5)
    entries.append(entry)

processor_speed_entry = entries[0]
ram_size_entry = entries[1]
storage_capacity_entry = entries[2]

# Predict button
predict_button = tk.Button(root, text="Predict Price", command=predict_price)
predict_button.grid(row=3, column=0, columnspan=2, padx=10, pady=10)

# Reset button
reset_button = tk.Button(root, text="Reset", command=reset_entries)
reset_button.grid(row=4, column=0, columnspan=2, padx=10, pady=10)

# Predicted price label
predicted_price_label = tk.Label(root, text="")
predicted_price_label.grid(row=5, column=0, columnspan=2, padx=10, pady=5)

root.mainloop()
