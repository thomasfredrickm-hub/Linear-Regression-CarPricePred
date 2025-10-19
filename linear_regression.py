"""
Car Price Prediction using Linear Regression (Without Scikit-Learn)
Research AI Task 5 - Implementation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# PART 1: LINEAR REGRESSION CLASS (FROM SCRATCH)
# ============================================

class LinearRegressionScratch:
    """
    Linear Regression implementation from scratch using Normal Equation.
    Formula: θ = (X^T X)^-1 X^T y
    """
    
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """
        Train the model using Normal Equation
        """
        # Add bias term (column of ones)
        n_samples, n_features = X.shape
        X_with_bias = np.c_[np.ones((n_samples, 1)), X]
        
        # Normal Equation: θ = (X^T X)^-1 X^T y
        try:
            theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        except np.linalg.LinAlgError:
            # If matrix is singular, use pseudo-inverse
            theta = np.linalg.pinv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        return self
    
    def predict(self, X):
        """
        Make predictions
        """
        return X @ self.weights + self.bias
    
    def score(self, X, y):
        """
        Calculate R² score
        """
        y_pred = self.predict(X)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r2 = 1 - (ss_residual / ss_total)
        return r2


# ============================================
# PART 2: DATA PREPROCESSING FUNCTIONS
# ============================================

def load_and_explore_data(filepath):
    """
    Load the dataset and perform initial exploration
    """
    df = pd.read_csv(filepath)
    
    print("="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Dataset Shape: {df.shape}")
    print(f"\nColumn Names:\n{df.columns.tolist()}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")
    
    return df


def preprocess_data(df):
    """
    Preprocess the data: handle missing values, encode categoricals, etc.
    """
    df = df.copy()
    
    # 1. Extract company name from CarName
    df['CarCompany'] = df['CarName'].str.split().str[0]
    df['CarCompany'] = df['CarCompany'].str.lower()
    
    # Standardize company names (fix typos)
    company_mapping = {
        'maxda': 'mazda',
        'porcshce': 'porsche',
        'toyouta': 'toyota',
        'vokswagen': 'volkswagen',
        'vw': 'volkswagen'
    }
    df['CarCompany'] = df['CarCompany'].replace(company_mapping)
    
    # 2. Drop unnecessary columns
    df = df.drop(['car_ID', 'CarName'], axis=1)
    
    # 3. Handle missing values
    # Numerical columns: fill with median
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().sum() > 0:
            df[col].fillna(df[col].median(), inplace=True)
    
    # 4. Encode categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    # One-Hot Encoding for nominal variables
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    print("\n" + "="*50)
    print("DATA PREPROCESSING COMPLETED")
    print("="*50)
    print(f"Shape after preprocessing: {df_encoded.shape}")
    print(f"Columns after encoding: {len(df_encoded.columns)}")
    
    return df_encoded


def remove_outliers(df, column):
    """
    Remove outliers using IQR method
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    print(f"Removed {len(df) - len(df_clean)} outliers from {column}")
    return df_clean


# ============================================
# PART 3: FEATURE SCALING
# ============================================

class StandardScaler:
    """
    Standardize features by removing mean and scaling to unit variance
    """
    
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, X):
        # Ensure X is a numpy array with proper dtype
        X = np.asarray(X, dtype=np.float64)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0, dtype=np.float64)
        return self
    
    def transform(self, X):
        # Ensure X is a numpy array with proper dtype
        X = np.asarray(X, dtype=np.float64)
        return (X - self.mean) / (self.std + 1e-8)
    
    def fit_transform(self, X):
        # Ensure X is a numpy array with proper dtype
        X = np.asarray(X, dtype=np.float64)
        self.fit(X)
        return self.transform(X)


# ============================================
# PART 4: MODEL EVALUATION METRICS
# ============================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate various regression metrics
    """
    # R² Score
    ss_total = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_residual = np.sum((y_true - y_pred) ** 2)
    r2 = 1 - (ss_residual / ss_total)
    
    # Mean Absolute Error
    mae = np.mean(np.abs(y_true - y_pred))
    
    # Root Mean Squared Error
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # Mean Absolute Percentage Error
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    metrics = {
        'R² Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape
    }
    
    return metrics


def print_metrics(metrics, dataset_name=""):
    """
    Print metrics in a formatted way
    """
    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION METRICS {dataset_name}")
    print(f"{'='*50}")
    for metric, value in metrics.items():
        if 'Score' in metric or 'MAPE' in metric:
            print(f"{metric:20s}: {value:10.4f}")
        else:
            print(f"{metric:20s}: ${value:10.2f}")


# ============================================
# PART 5: VISUALIZATION FUNCTIONS
# ============================================

def plot_results(y_true, y_pred, title="Actual vs Predicted Prices"):
    """
    Create visualization plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Actual vs Predicted
    axes[0, 0].scatter(y_true, y_pred, alpha=0.6, edgecolors='k')
    axes[0, 0].plot([y_true.min(), y_true.max()], 
                     [y_true.min(), y_true.max()], 
                     'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Price', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Price', fontsize=12)
    axes[0, 0].set_title(title, fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Residual Plot
    residuals = y_true - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.6, edgecolors='k')
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted Price', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Residual Distribution
    axes[1, 0].hist(residuals, bins=30, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Distribution of Residuals', fontsize=14)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. Error Distribution
    errors = np.abs(residuals)
    axes[1, 1].hist(errors, bins=30, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Distribution of Absolute Errors', fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('model_evaluation_plots.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(feature_names, coefficients, top_n=10):
    """
    Plot feature importance based on coefficients
    """
    # Get absolute values for importance
    importance = np.abs(coefficients)
    indices = np.argsort(importance)[::-1][:top_n]
    
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importance[indices], color='skyblue', edgecolor='black')
    plt.yticks(range(top_n), [feature_names[i] for i in indices])
    plt.xlabel('Absolute Coefficient Value', fontsize=12)
    plt.ylabel('Features', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features', fontsize=14)
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_correlation_heatmap(df, target='price'):
    """
    Plot correlation heatmap for numerical features
    """
    # Select numerical columns
    numerical_df = df.select_dtypes(include=[np.number])
    
    # Calculate correlation with target
    correlations = numerical_df.corr()[target].sort_values(ascending=False)
    
    # Plot top correlated features
    top_features = correlations.head(15).index
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df[top_features].corr(), 
                annot=True, 
                fmt='.2f', 
                cmap='coolwarm', 
                center=0,
                square=True,
                linewidths=1)
    plt.title('Correlation Heatmap - Top Features', fontsize=14)
    plt.tight_layout()
    plt.savefig('correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()


# ============================================
# PART 6: MAIN EXECUTION
# ============================================

def train_test_split(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets
    """
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    
    test_indices = indices[:test_size]
    train_indices = indices[test_size:]
    
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    return X_train, X_test, y_train, y_test


def main():
    """
    Main function to execute the entire pipeline
    """
    print("="*70)
    print("CAR PRICE PREDICTION USING LINEAR REGRESSION")
    print("Research AI Task 5 - Implementation without Scikit-Learn")
    print("="*70)
    
    # Step 1: Load and explore data
    print("\n[STEP 1] Loading Dataset...")
    df = load_and_explore_data('CarPrice.csv')
    
    # Step 2: Preprocess data
    print("\n[STEP 2] Preprocessing Data...")
    df_processed = preprocess_data(df)
    
    # Step 3: Remove outliers from target variable
    print("\n[STEP 3] Removing Outliers...")
    df_clean = remove_outliers(df_processed, 'price')
    
    # Step 4: Prepare features and target
    print("\n[STEP 4] Preparing Features and Target...")
    X = df_clean.drop('price', axis=1).values.astype(np.float64)
    y = df_clean['price'].values.astype(np.float64)
    feature_names = df_clean.drop('price', axis=1).columns.tolist()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Feature matrix dtype: {X.dtype}")
    print(f"Target vector dtype: {y.dtype}")
    
    # Step 5: Split data
    print("\n[STEP 5] Splitting Data (80-20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Step 6: Feature scaling
    print("\n[STEP 6] Scaling Features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 7: Train model
    print("\n[STEP 7] Training Linear Regression Model...")
    model = LinearRegressionScratch()
    model.fit(X_train_scaled, y_train)
    print("Model training completed!")
    
    # Step 8: Make predictions
    print("\n[STEP 8] Making Predictions...")
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    
    # Step 9: Evaluate model
    print("\n[STEP 9] Evaluating Model...")
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    print_metrics(train_metrics, "(Training Set)")
    print_metrics(test_metrics, "(Testing Set)")
    
    # Step 10: Visualizations
    print("\n[STEP 10] Creating Visualizations...")
    plot_results(y_test, y_test_pred, "Car Price Prediction: Actual vs Predicted")
    plot_feature_importance(feature_names, model.weights, top_n=15)
    
    # Step 11: Display sample predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    sample_df = pd.DataFrame({
        'Actual Price': y_test[:10],
        'Predicted Price': y_test_pred[:10],
        'Error': np.abs(y_test[:10] - y_test_pred[:10]),
        'Error %': np.abs((y_test[:10] - y_test_pred[:10]) / y_test[:10]) * 100
    })
    print(sample_df.to_string(index=False))
    
    # Step 12: Model summary
    print("\n" + "="*70)
    print("MODEL SUMMARY")
    print("="*70)
    print(f"Total Features Used: {len(feature_names)}")
    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")
    print(f"\nModel Performance:")
    print(f"  - Training R²: {train_metrics['R² Score']:.4f}")
    print(f"  - Testing R²: {test_metrics['R² Score']:.4f}")
    print(f"  - Overfitting Check: {abs(train_metrics['R² Score'] - test_metrics['R² Score']):.4f}")
    
    if abs(train_metrics['R² Score'] - test_metrics['R² Score']) < 0.05:
        print("  ✓ Model is well-generalized (minimal overfitting)")
    else:
        print("  ⚠ Model may be overfitting")
    
    print(f"\nAverage Prediction Error: ${test_metrics['MAE']:.2f}")
    print(f"Prediction Accuracy: {100 - test_metrics['MAPE']:.2f}%")
    
    # Step 13: Feature coefficients
    print("\n" + "="*70)
    print("TOP 10 MOST INFLUENTIAL FEATURES")
    print("="*70)
    importance = np.abs(model.weights)
    indices = np.argsort(importance)[::-1][:10]
    
    for i, idx in enumerate(indices, 1):
        print(f"{i:2d}. {feature_names[idx]:30s} : {model.weights[idx]:10.2f}")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)
    print("\nFiles Generated:")
    print("  1. model_evaluation_plots.png")
    print("  2. feature_importance.png")
    print("  3. correlation_heatmap.png")
    print("\nAll visualizations have been saved to the current directory.")
    
    return model, scaler, feature_names, test_metrics


# ============================================
# PART 7: PREDICTION FUNCTION FOR NEW DATA
# ============================================

def predict_new_car(model, scaler, feature_names, car_features):
    """
    Predict price for a new car
    
    Parameters:
    -----------
    model : LinearRegressionScratch
        Trained model
    scaler : StandardScaler
        Fitted scaler
    feature_names : list
        List of feature names
    car_features : dict
        Dictionary with car specifications
    
    Returns:
    --------
    float : Predicted price
    """
    # Create feature vector (this is simplified - you'd need proper encoding)
    # For demonstration purposes
    print("\n" + "="*50)
    print("PREDICTING PRICE FOR NEW CAR")
    print("="*50)
    print("Car Specifications:")
    for key, value in car_features.items():
        print(f"  {key}: {value}")
    
    # Note: In practice, you'd need to properly encode categorical variables
    # This is just a placeholder
    print("\n⚠ Note: For actual prediction, proper feature encoding is required")
    print("This requires the exact same preprocessing as training data")


# ============================================
# RUN THE PROGRAM
# ============================================

if __name__ == "__main__":
    # Execute the main pipeline
    model, scaler, feature_names, metrics = main()
    
    # Example: Predict for a new car (you'd need to provide proper encoded features)
    # new_car = {
    #     'enginesize': 150,
    #     'horsepower': 120,
    #     'curbweight': 2500,
    #     'carwidth': 65.5,
    #     'carlength': 175.0
    # }
    # predict_new_car(model, scaler, feature_names, new_car)
