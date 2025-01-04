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

1. 