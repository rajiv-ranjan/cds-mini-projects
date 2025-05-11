# Lessons Learnt

## Data Preprocessing

1. What data type should be used for categorical and continuous data?
    1. categorical: int64
    1. continuous data: float64
1. When to use `MinMaxScaler` and `StandardScaler`?

```md
1. Use MinMaxScaler when:
    1. You know the bounds of your features should be preserved
    2. Working with neural networks
    3. Features are already normally distributed
    4. Dealing with image processing

2. Use StandardScaler when:
    1. You're unsure about the distribution of your data
    2. Using algorithms that assume normal distribution
    3. Working with linear models
    4. Features have very different scales and outliers

3. Special Cases:
    1. For tree-based models (Random Forest, XGBoost): Usually no scaling needed
    2. For text data: Special scalers like TF-IDF are often better
    3. For PCA: Always use StandardScaler

Remember: Always scale features before splitting into train/test sets to avoid data leakage, and apply the same scaling parameters to both sets.
```

## Scaling & Encoding

1. Should we apply scalar and encoding to both feature and target before training in linear regression models? Will the answer change for logistic regression?

```md
Scaling for Linear Regression:
- Features: Yes, scaling is generally important for features because:
  1. It helps when features are on different scales (e.g., age vs. income)
  2. It can speed up gradient descent
  3. It prevents features with larger scales from dominating the model
- Target variable: Scaling is optional and depends on your needs:
  1. If you care about the interpretability of the predictions in the original scale, don't scale the target
  2. If you're more concerned with model performance or training stability, scaling can help
  3. You'll need to inverse transform predictions to get back to the original scale

Scaling for Logistic Regression:
- Features: Yes, scaling is still important for the same reasons as linear regression
- Target variable: No, don't scale the target because:
  1. Logistic regression targets are binary (0/1) or categorical
  2. The model is specifically designed to handle unscaled binary/categorical outcomes
  3. The sigmoid function naturally maps outputs to probabilities between 0 and 1

As for encoding:
- Features: Both types of regression require encoding for categorical variables (e.g., one-hot encoding, label encoding)
- Target: 
  - Linear regression: No encoding needed for numeric targets
  - Logistic regression: Categorical targets should be encoded (usually label encoding for binary, one-hot for multiclass)

```