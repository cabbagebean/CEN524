
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression

# ======================
# Activity 1: Synthetic Data & Normalization
# ======================
def activity_1_generate_synthetic_data():
    print("\n--- Activity 1: Generate and Normalize Synthetic Data ---")
    np.random.seed(42)
    n_samples = 100
    tickets_handled = np.random.randint(5, 50, size=n_samples)
    response_time = np.random.normal(loc=30, scale=10, size=n_samples)
    satisfaction_score = (
        5 - (response_time / 50) + (tickets_handled / 100) + np.random.normal(0, 0.2, n_samples)
    ).clip(1, 5)

    # Normalize the tickets_handled (1 feature)
    scaler = StandardScaler()
    X = scaler.fit_transform(tickets_handled.reshape(-1, 1)).flatten()
    y = satisfaction_score

    return X, y
#Q:How does normalization affect the feature values?
#ANS: Normalization rescales features to have zero mean and unit variance, ensuring fair contribution and faster model convergence.



# ======================
# Activity 2: Cost Function Calculation
# ======================
def activity_2_cost_functions(X, y):
    print("\n--- Activity 2: Cost Function Calculation ---")
    params = [(1.5, 0.5), (0.8, 1.2), (2.0, -0.3)]
    for w, b in params:
        y_pred = w * X + b
        mse = np.mean((y - y_pred) ** 2)
        mae = np.mean(np.abs(y - y_pred))
        print(f"w={w}, b={b} → MSE={mse:.4f}, MAE={mae:.4f}")

#Q:Why does MSE penalize larger errors more than MAE?
#ANS: MSE squares the errors, so larger errors grow exponentially, while MAE treats all errors linearly.



# ======================
# Activity 3: Gradient Descent on Synthetic Data
# ======================
def activity_3_gradient_descent(X, y):
    print("\n--- Activity 3: Gradient Descent (Synthetic 1 Feature) ---")
    w = 0.0
    b = 0.0
    alpha = 0.01
    iterations = 100
    m = len(X)
    mse_history = []

    for _ in range(iterations):
        y_pred = w * X + b
        error = y_pred - y
        dw = (2 / m) * np.dot(error, X)
        db = (2 / m) * np.sum(error)
        w -= alpha * dw
        b -= alpha * db
        mse_history.append(np.mean(error ** 2))

    print(f"✅ Optimized w: {w:.4f}, b: {b:.4f}, Final MSE: {mse_history[-1]:.4f}")

    plt.plot(mse_history)
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.title("MSE over Iterations (Gradient Descent)")
    plt.grid(True)
    plt.show()

#Q:How does the learning rate affect convergence?
#ANS: A small learning rate leads to slow convergence, while a large one can cause overshooting or divergence.

# ======================
# Activity 4: Real Dataset Preprocessing & Model Evaluation
# ======================
def activity_4_sklearn_model():
    print("\n--- Activity 4 (Fast Version): Linear Regression with scikit-learn ---")

    df = pd.read_csv("customer_support_tickets.csv")
    df = df.dropna(subset=['Customer Satisfaction Rating'])
    df['First_Response_Hour'] = pd.to_datetime(df['First Response Time'], errors='coerce').dt.hour

    X = df[['Customer Age', 'Ticket Priority', 'Ticket Channel', 'Ticket Type', 'First_Response_Hour']]
    y = df['Customer Satisfaction Rating']

    numeric_features = ['Customer Age', 'First_Response_Hour']
    categorical_features = ['Ticket Priority', 'Ticket Channel', 'Ticket Type']

    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(drop='first'), categorical_features)
    ])

    X_processed = preprocessor.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"✅ Scikit-learn Test MSE: {mse:.4f}")
    print(f"✅ Scikit-learn Test MAE: {mae:.4f}")
    print(f"Intercept (b): {model.intercept_:.4f}")
    print(f"Coefficients (w): {model.coef_}")
    

#Q:Why might the model perform differently on real vs. synthetic data?
#ANS:Real data is noisier and more complex, while synthetic data is cleaner and more predictable.


# ======================
# Run All Activities in Order
# ======================
if __name__ == "__main__":
    X_synthetic, y_synthetic = activity_1_generate_synthetic_data()
    activity_2_cost_functions(X_synthetic, y_synthetic)
    activity_3_gradient_descent(X_synthetic, y_synthetic)
    activity_4_sklearn_model()

#Q:How does the choice of cost function (MSE vs. MAE) affect optimization?
#ANS:MSE penalizes large errors more (smoother gradients), while MAE is more robust to outliers but harder to optimize.

#Q:What challenges arise when scaling to multiple features?
#ANS:Scaling increases computational complexity, risks overfitting, and requires feature normalization to ensure fair learning.

#Q:How does gradient descent compare to scikit-learn's built-in linear regression?
#ANS:Gradient descent is iterative and flexible, while scikit-learn's LinearRegression uses a fast, exact, closed-form solution.
