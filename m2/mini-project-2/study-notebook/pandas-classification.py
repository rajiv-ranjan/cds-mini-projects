import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import matplotlib.pyplot as plt

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics using pandas Series operations
    """
    # Convert inputs to pandas Series if they aren't already
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    
    # Calculate metrics using pandas operations
    TP = sum((y_true == 1) & (y_pred == 1))
    TN = sum((y_true == 0) & (y_pred == 0))
    FP = sum((y_true == 0) & (y_pred == 1))
    FN = sum((y_true == 1) & (y_pred == 0))
    
    # Calculate derived metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return pd.Series({
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    })

# Create imbalanced dataset using pandas
np.random.seed(42)
n_samples = 1000
n_class_0 = int(0.93 * n_samples)
n_class_1 = n_samples - n_class_0

# Create DataFrame with features and target
df = pd.DataFrame({
    'feature1': np.concatenate([
        np.random.normal(0, 1, n_class_0),
        np.random.normal(2, 1, n_class_1)
    ]),
    'feature2': np.concatenate([
        np.random.normal(0, 1, n_class_0),
        np.random.normal(2, 1, n_class_1)
    ]),
    'target': np.concatenate([
        np.zeros(n_class_0),
        np.ones(n_class_1)
    ])
})

# Split features and target
X = df[['feature1', 'feature2']]
y = df['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model without weights
model_no_weights = LogisticRegression(random_state=42)
model_no_weights.fit(X_train, y_train)
y_pred_no_weights = model_no_weights.predict(X_test)

# Calculate class weights using pandas
class_weights = dict(
    df['target'].value_counts(normalize=True).apply(lambda x: n_samples/(2*n_samples*x))
)

# Train model with weights
model_with_weights = LogisticRegression(random_state=42, class_weight=class_weights)
model_with_weights.fit(X_train, y_train)
y_pred_with_weights = model_with_weights.predict(X_test)

# Calculate metrics for both models
metrics_no_weights = calculate_metrics(y_test, y_pred_no_weights)
metrics_with_weights = calculate_metrics(y_test, y_pred_with_weights)

# Create DataFrame with all metrics
metrics_df = pd.DataFrame({
    'Without Weights': metrics_no_weights,
    'With Weights': metrics_with_weights
})

# Create confusion matrices using pandas crosstab
conf_matrix_no_weights = pd.crosstab(
    y_test, pd.Series(y_pred_no_weights),
    rownames=['Actual'],
    colnames=['Predicted']
)

conf_matrix_with_weights = pd.crosstab(
    y_test, pd.Series(y_pred_with_weights),
    rownames=['Actual'],
    colnames=['Predicted']
)

# Print results
print("Class Distribution in Training Data:")
print(df['target'].value_counts(normalize=True).mul(100).round(2), "\n")

print("Class Weights Used:")
print(pd.Series(class_weights), "\n")

print("Model Performance Metrics:")
print(metrics_df.round(3), "\n")

print("\nConfusion Matrix - Without Weights:")
print(conf_matrix_no_weights)

print("\nConfusion Matrix - With Weights:")
print(conf_matrix_with_weights)

# Optional: Create visualizations
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot feature distribution
sns.scatterplot(
    data=df,
    x='feature1',
    y='feature2',
    hue='target',
    ax=ax1,
    alpha=0.6
)
ax1.set_title('Feature Distribution')

# Plot metrics comparison
metrics_df.loc[['accuracy', 'precision', 'recall', 'f1_score']].plot(
    kind='bar',
    ax=ax2
)
ax2.set_title('Model Metrics Comparison')
ax2.set_ylabel('Score')
plt.tight_layout()
