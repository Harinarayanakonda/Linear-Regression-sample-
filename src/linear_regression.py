# linear_regression_module.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def main():
    # Data
    x = [151, 174, 138, 186, 128, 136, 179, 163, 152, 131]
    y = [63, 81, 56, 91, 47, 57, 76, 72, 62, 48]

    # Convert to NumPy arrays and reshape x
    X = np.array(x).reshape(-1, 1)
    Y = np.array(y)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, Y)

    # Print model parameters
    print("Linear Regression Model:")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"Slope: {model.coef_[0]:.2f}")

    # Predict using the trained model
    Y_pred = model.predict(X)

    # Metrics
    print(f"Mean Squared Error: {mean_squared_error(Y, Y_pred):.2f}")
    print(f"R² Score: {r2_score(Y, Y_pred):.2f}")

    # Plot
    plt.scatter(X, Y, color='blue', label='Actual data')
    plt.plot(X, Y_pred, color='red', linewidth=2, label='Regression line')
    plt.title("Height vs. Weight Linear Regression")
    plt.xlabel("Height")
    plt.ylabel("Weight")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Predict for a new height
    new_height = 320
    predicted_weight = model.predict([[new_height]])
    print(f"Predicted weight for height {new_height} is {predicted_weight[0]:.2f}")

if __name__ == "__main__":
    main()
