# Statistical Methods

## Data Preparation

### Missing Data Treatment
- **Method**: Multiple Imputation by Chained Equations (MICE)
- **Variables**: Primarily financial transparency (10.3% missing)
- **Iterations**: 10 iterations, 5 imputed datasets
- **Validation**: Convergence diagnostics and distribution comparisons

### Normalization
- **Robust Scaling**: Applied to all risk variables for statistical analysis
- **Formula**: (x - median) / IQR
- **Rationale**: Handles outliers and wide ranges better than z-score normalization

## Analysis Methods

### Descriptive Analysis
- **Risk Categorization**: Quartile-based (Low/Medium/High/Very High)
- **Regional Comparisons**: One-way ANOVA with Tukey HSD post-hoc
- **Normality Testing**: Shapiro-Wilk, Anderson-Darling, Kolmogorov-Smirnov

### Multivariate Analysis
- **Correlation**: Pearson with partial correlation controlling for region
- **PCA**: Standardized variables, varimax rotation, Kaiser criterion
- **Feature Importance**: Random Forest with 500 trees, 10-fold CV

### Regulatory Impact
- **Effect Sizes**: Cohen's d with 95% confidence intervals
- **Statistical Tests**: Independent t-tests with Bonferroni correction
- **Interaction Analysis**: Two-way ANOVA for FATF × other sanctions

### Predictive Modeling
- **Algorithms**: Linear Regression, Ridge, Random Forest, XGBoost
- **Validation**: 10-fold cross-validation, 80/20 train-test split
- **Metrics**: R², RMSE, MAE with prediction intervals
- **Decision Trees**: Binary classification with interpretable depth limits

## Statistical Assumptions

### Parametric Tests
- **Normality**: Assessed via multiple tests and Q-Q plots
- **Homoscedasticity**: Levene's test for equality of variances
- **Independence**: Verified through residual analysis

### Effect Size Interpretation
- **Cohen's d**: Small (0.2), Medium (0.5), Large (0.8)
- **Eta-squared**: Small (0.01), Medium (0.06), Large (0.14)
- **R-squared**: Variance explained in regression models

## Multiple Comparisons
- **Bonferroni Correction**: Applied for 11 sanction comparisons
- **FDR Control**: Used for correlation significance testing
- **Family-wise Error Rate**: Maintained at α = 0.05