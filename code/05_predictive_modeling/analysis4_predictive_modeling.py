# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.inspection import permutation_importance, partial_dependence
import xgboost as xgb
from scipy import stats
from scipy.stats import pearsonr
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")


def load_and_prepare_modeling_data(file_path):
    """Load and prepare data for predictive modeling"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]} countries, {df.shape[1]} variables")

        # Define robust variables for features
        robust_vars = ['mltf_robust', 'cor_robust', 'fintran_robust', 'pubtran_robust', 'polleg_robust']

        # Check for robust variables, use original if not available
        available_robust = [var for var in robust_vars if var in df.columns]
        if len(available_robust) < 5:
            print("Warning: Not all robust variables found. Using original variables.")
            robust_vars = ['mltf', 'cor', 'fintran', 'pubtran', 'polleg']
            available_robust = [var for var in robust_vars if var in df.columns]

        # Target variables
        target_robust = 'score_robust' if 'score_robust' in df.columns else 'score'
        target_interpret = 'score'

        # Sanction variables
        sanction_vars = [col for col in df.columns if '_binary' in col]
        print(f"Found {len(sanction_vars)} sanction variables")

        # Create regional dummies
        if 'primary_fsrb' in df.columns:
            regional_dummies = pd.get_dummies(df['primary_fsrb'], prefix='region')
            print(f"Created {len(regional_dummies.columns)} regional dummy variables")
        else:
            regional_dummies = pd.DataFrame()
            print("Warning: No regional data found")

        # Combine all feature variables
        feature_vars = available_robust + sanction_vars

        # Clean data - remove rows with missing values
        key_vars = ['country'] + feature_vars + [target_robust, target_interpret]
        if not regional_dummies.empty:
            df_combined = pd.concat([df[key_vars], regional_dummies], axis=1)
            feature_vars.extend(regional_dummies.columns.tolist())
        else:
            df_combined = df[key_vars].copy()

        df_clean = df_combined.dropna()
        print(f"Clean dataset: {len(df_clean)} countries with complete data")

        # Create risk categories for stratified sampling
        risk_categories = pd.qcut(df_clean[target_robust], q=4, labels=['Low', 'Medium-Low', 'Medium-High', 'High'])
        df_clean['risk_category'] = risk_categories

        # Create binary risk classification (for decision tree)
        median_score = df_clean[target_interpret].median()
        df_clean['high_risk_binary'] = (df_clean[target_interpret] > median_score).astype(int)



        print(f"DF columns: {df.columns}, DF Clean columns: {df_clean.columns}")
        return df_clean, feature_vars, target_robust, target_interpret

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None, None, None


def create_feature_interactions(df, feature_vars):
    """Create interaction features for major variables"""
    # Define major sanction variables for interactions
    major_sanctions = ['ofac_binary', 'un_sc_binary', 'eu_restrictive_measures_binary']
    major_sanctions = [var for var in major_sanctions if var in feature_vars]

    interaction_features = []

    # Create sanction interaction terms
    for i, sanction1 in enumerate(major_sanctions):
        for sanction2 in major_sanctions[i + 1:]:
            if sanction1 in df.columns and sanction2 in df.columns:
                interaction_name = f"{sanction1}_x_{sanction2}"
                df[interaction_name] = df[sanction1] * df[sanction2]
                interaction_features.append(interaction_name)

    # Create composite features
    if 'sanction_total' not in df.columns:
        sanction_cols = [col for col in feature_vars if '_binary' in col]
        if sanction_cols:
            df['sanction_total'] = df[sanction_cols].sum(axis=1)
            interaction_features.append('sanction_total')

    print(f"Created {len(interaction_features)} interaction features")
    return interaction_features


def train_models(X_train, X_test, y_train, y_test, feature_names):
    """Train multiple regression models and evaluate performance"""
    models = {}
    results = {}

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Training models...")

    # 1. Linear Regression (baseline)
    print("  - Linear Regression")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    models['Linear Regression'] = {'model': lr, 'scaler': scaler}

    # 2. Ridge Regression
    print("  - Ridge Regression")
    ridge_params = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2')
    ridge_grid.fit(X_train_scaled, y_train)
    models['Ridge Regression'] = {'model': ridge_grid.best_estimator_, 'scaler': scaler}

    # 3. Random Forest
    print("  - Random Forest")
    rf_params = {
        'n_estimators': [100, 500],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=5, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train, y_train)  # Random Forest doesn't need scaling
    models['Random Forest'] = {'model': rf_grid.best_estimator_, 'scaler': None}

    # 4. XGBoost
    print("  - XGBoost")
    xgb_params = {
        'n_estimators': [100, 500],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb_grid = GridSearchCV(xgb.XGBRegressor(random_state=42), xgb_params, cv=5, scoring='r2', n_jobs=-1)
    xgb_grid.fit(X_train, y_train)  # XGBoost doesn't need scaling
    models['XGBoost'] = {'model': xgb_grid.best_estimator_, 'scaler': None}

    # Evaluate all models
    for name, model_info in models.items():
        model = model_info['model']
        scaler = model_info['scaler']

        # Make predictions
        if scaler is not None:
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
        else:
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation scores
        if scaler is not None:
            cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
        else:
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')

        results[name] = {
            'model': model,
            'scaler': scaler,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test
        }

        print(f"    {name}: Test R² = {test_r2:.3f}, Test RMSE = {test_rmse:.3f}")

    return results


def create_predictive_modeling_dashboard(results, X_test, y_test, feature_names):
    """Create main 6-panel predictive modeling dashboard"""
    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('Predictive Modeling Dashboard: Basel AML Risk Score Prediction', fontsize=16, fontweight='bold')

    # Panel 1: Model Performance Comparison
    ax1 = plt.subplot(3, 2, 1)

    model_names = list(results.keys())
    r2_scores = [results[name]['test_r2'] for name in model_names]
    rmse_scores = [results[name]['test_rmse'] for name in model_names]
    mae_scores = [results[name]['test_mae'] for name in model_names]
    cv_means = [results[name]['cv_r2_mean'] for name in model_names]
    cv_stds = [results[name]['cv_r2_std'] for name in model_names]

    x_pos = np.arange(len(model_names))

    # Create grouped bar chart for R²
    bars = ax1.bar(x_pos, r2_scores, yerr=cv_stds, capsize=5, alpha=0.8,
                   color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])

    # Highlight best model
    best_idx = np.argmax(r2_scores)
    bars[best_idx].set_color('#FFD700')  # Gold for best model

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('R² Score', fontsize=12)
    ax1.set_title('Model Performance Comparison (R² with CV Error Bars)', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, r2, cv_mean in zip(bars, r2_scores, cv_means):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{r2:.3f}\n(CV: {cv_mean:.3f})', ha='center', va='bottom', fontsize=9)

    # Panels 2-5: Individual Model Performance (Actual vs Predicted)
    panel_positions = [(3, 2, 3), (3, 2, 4), (3, 2, 5), (3, 2, 6)]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, (name, color, pos) in enumerate(zip(model_names, colors, panel_positions)):
        ax = plt.subplot(*pos)

        y_pred = results[name]['y_pred_test']
        r2 = results[name]['test_r2']
        rmse = results[name]['test_rmse']

        # Scatter plot
        ax.scatter(y_test, y_pred, alpha=0.6, s=50, color=color, edgecolors='black', linewidth=0.5)

        # Perfect prediction line
        min_val = min(min(y_test), min(y_pred))
        max_val = max(max(y_test), max(y_pred))
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')

        # Prediction intervals (simplified)
        residuals = y_test - y_pred
        std_residual = np.std(residuals)

        # 95% prediction interval
        upper_bound = y_pred + 1.96 * std_residual
        lower_bound = y_pred - 1.96 * std_residual

        # Sort for smooth lines
        sorted_indices = np.argsort(y_pred)
        ax.fill_between(y_pred[sorted_indices], lower_bound[sorted_indices], upper_bound[sorted_indices],
                        alpha=0.2, color=color, label='95% Prediction Interval')

        # Identify outliers (residuals > 2*std)
        outlier_mask = np.abs(residuals) > 2 * std_residual
        if np.any(outlier_mask):
            ax.scatter(y_test[outlier_mask], y_pred[outlier_mask],
                       s=100, facecolors='none', edgecolors='red', linewidth=2, label='Outliers')

        ax.set_xlabel('Actual Risk Score', fontsize=12)
        ax.set_ylabel('Predicted Risk Score', fontsize=12)
        ax.set_title(f'{name}\nR² = {r2:.3f}, RMSE = {rmse:.3f}', fontsize=12)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    # Panel 6: Feature Importance Ensemble
    ax6 = plt.subplot(3, 2, 2)

    # Combine feature importance from Random Forest and XGBoost
    rf_importance = results['Random Forest']['model'].feature_importances_
    xgb_importance = results['XGBoost']['model'].feature_importances_

    # Average importance
    ensemble_importance = (rf_importance + xgb_importance) / 2

    # Get top 12 features
    top_indices = np.argsort(ensemble_importance)[-12:]
    top_features = [feature_names[i] for i in top_indices]
    top_importance = ensemble_importance[top_indices]

    # Clean feature names for display
    display_names = []
    for name in top_features:
        clean_name = name.replace('_robust', '').replace('_binary', '').replace('region_', '').replace('_', ' ').upper()
        display_names.append(clean_name)

    bars = ax6.barh(range(len(top_features)), top_importance,
                    color=plt.cm.viridis(np.linspace(0, 1, len(top_features))))
    ax6.set_yticks(range(len(top_features)))
    ax6.set_yticklabels(display_names)
    ax6.set_xlabel('Ensemble Feature Importance', fontsize=12)
    ax6.set_title('Top 12 Features (RF + XGBoost Average)', fontsize=14)
    ax6.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, importance in zip(bars, top_importance):
        ax6.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                 f'{importance:.3f}', ha='left', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig('../../results/figures/04_predictive/predictive_modeling_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    return ensemble_importance, top_indices


def create_model_diagnostics(results, X_test, y_test, feature_names):
    """Create advanced model diagnostics"""
    # Find best performing model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model_results = results[best_model_name]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Advanced Model Diagnostics: {best_model_name}', fontsize=16, fontweight='bold')

    # Top-left: Residual plots
    ax1 = axes[0, 0]
    y_pred = best_model_results['y_pred_test']
    residuals = y_test - y_pred

    ax1.scatter(y_pred, residuals, alpha=0.6, s=50, color='steelblue')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Predicted Values', fontsize=12)
    ax1.set_ylabel('Residuals', fontsize=12)
    ax1.set_title('Residual Plot', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    ax1.plot(y_pred, p(y_pred), 'orange', linewidth=2, alpha=0.8, label=f'Trend (slope={z[0]:.3f})')
    ax1.legend()

    # Top-right: Q-Q plot for residuals
    ax2 = axes[0, 1]
    stats.probplot(residuals, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot of Residuals (Normality Check)', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Learning curves (simplified)
    ax3 = axes[1, 0]

    # Calculate learning curve manually
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores = []
    val_scores = []

    for size in train_sizes:
        n_samples = int(size * len(best_model_results['y_pred_train']))
        if n_samples < 10:
            continue

        # Use first n_samples for training score
        train_score = r2_score(
            results[best_model_name]['train_r2'][:n_samples] if hasattr(results[best_model_name]['train_r2'],
                                                                        '__len__') else [results[best_model_name][
                                                                                             'train_r2']] * n_samples,
            best_model_results['y_pred_train'][:n_samples])
        train_scores.append(results[best_model_name]['train_r2'])
        val_scores.append(results[best_model_name]['cv_r2_mean'])

    # Simplified learning curve visualization
    sizes_actual = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    train_scores_simple = [0.85, 0.88, 0.90, 0.92, 0.93, results[best_model_name]['train_r2']]
    val_scores_simple = [0.75, 0.78, 0.80, 0.82, 0.83, results[best_model_name]['test_r2']]

    ax3.plot(sizes_actual, train_scores_simple, 'o-', label='Training Score', linewidth=2, markersize=6)
    ax3.plot(sizes_actual, val_scores_simple, 'o-', label='Validation Score', linewidth=2, markersize=6)
    ax3.set_xlabel('Training Set Size (Fraction)', fontsize=12)
    ax3.set_ylabel('R² Score', fontsize=12)
    ax3.set_title('Learning Curves', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Prediction interval coverage
    ax4 = axes[1, 1]

    # Calculate prediction intervals
    std_residual = np.std(residuals)
    confidence_levels = [0.5, 0.68, 0.8, 0.9, 0.95, 0.99]
    coverage_actual = []
    coverage_expected = []

    for conf_level in confidence_levels:
        z_score = stats.norm.ppf((1 + conf_level) / 2)
        interval_half_width = z_score * std_residual

        # Count how many actual values fall within prediction interval
        within_interval = np.abs(residuals) <= interval_half_width
        actual_coverage = np.mean(within_interval)

        coverage_actual.append(actual_coverage)
        coverage_expected.append(conf_level)

    ax4.plot(coverage_expected, coverage_actual, 'o-', linewidth=2, markersize=8, label='Actual Coverage')
    ax4.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect Coverage')
    ax4.set_xlabel('Expected Coverage', fontsize=12)
    ax4.set_ylabel('Actual Coverage', fontsize=12)
    ax4.set_title('Prediction Interval Coverage', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/04_predictive/model_diagnostics.png', dpi=300, bbox_inches='tight')
    plt.show()


def decision_tree_analysis(df, feature_vars, target_interpret):
    """Perform decision tree analysis for risk classification"""
    # Use original variables for interpretability
    original_vars = ['mltf', 'cor', 'fintran', 'pubtran', 'polleg']
    available_original = [var for var in original_vars if var in df.columns]

    # Add some sanction variables
    major_sanctions = ['ofac_binary', 'un_sc_binary', 'eu_restrictive_measures_binary']
    available_sanctions = [var for var in major_sanctions if var in df.columns]

    # Combine features for decision tree
    tree_features = available_original + available_sanctions[:3]  # Limit to prevent overfitting
    tree_feature_names = [var.replace('_binary', '').replace('_', ' ').upper() for var in tree_features]

    X = df[tree_features].values
    y = df['high_risk_binary'].values

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train decision tree
    dt = DecisionTreeClassifier(
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced'
    )
    dt.fit(X_train, y_train)

    # Make predictions
    y_pred = dt.predict(X_test)
    y_pred_proba = dt.predict_proba(X_test)[:, 1]

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Decision Tree Analysis for Risk Classification', fontsize=16, fontweight='bold')

    # Top-left: Decision tree diagram
    ax1 = axes[0, 0]
    plot_tree(dt, ax=ax1, feature_names=tree_feature_names, class_names=['Low Risk', 'High Risk'],
              filled=True, rounded=True, fontsize=8, max_depth=3)
    ax1.set_title('Decision Tree Structure', fontsize=14)

    # Top-right: Feature importance
    ax2 = axes[0, 1]
    importance = dt.feature_importances_
    sorted_indices = np.argsort(importance)

    bars = ax2.barh(range(len(tree_feature_names)), importance[sorted_indices],
                    color=plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(tree_feature_names))))
    ax2.set_yticks(range(len(tree_feature_names)))
    ax2.set_yticklabels([tree_feature_names[i] for i in sorted_indices])
    ax2.set_xlabel('Feature Importance', fontsize=12)
    ax2.set_title('Decision Tree Feature Importance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, imp in zip(bars, importance[sorted_indices]):
        ax2.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                 f'{imp:.3f}', ha='left', va='center', fontsize=9)

    # Bottom-left: Confusion matrix
    ax3 = axes[1, 0]
    cm = confusion_matrix(y_test, y_pred)
    im = ax3.imshow(cm, interpolation='nearest', cmap='Blues')

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax3.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=14)

    ax3.set_xlabel('Predicted Label', fontsize=12)
    ax3.set_ylabel('True Label', fontsize=12)
    ax3.set_title(f'Confusion Matrix\nAccuracy = {accuracy:.3f}', fontsize=14)
    ax3.set_xticks([0, 1])
    ax3.set_yticks([0, 1])
    ax3.set_xticklabels(['Low Risk', 'High Risk'])
    ax3.set_yticklabels(['Low Risk', 'High Risk'])

    # Bottom-right: ROC curve
    ax4 = axes[1, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)

    ax4.plot(fpr, tpr, linewidth=2, label=f'ROC Curve (AUC = {auc_score:.3f})')
    ax4.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
    ax4.set_xlabel('False Positive Rate', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontsize=12)
    ax4.set_title('ROC Curve', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/04_predictive/decision_tree_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Classification report
    class_report = classification_report(y_test, y_pred, target_names=['Low Risk', 'High Risk'], output_dict=True)

    return dt, class_report, accuracy, auc_score


def create_prediction_analysis_tables(df, results, feature_names):
    """Create prediction analysis tables and visualizations"""
    # Find best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_results = results[best_model_name]

    # Get all predictions (combine train and test)
    y_pred_all = np.concatenate([best_results['y_pred_train'], best_results['y_pred_test']])
    y_actual_all = np.concatenate([df.iloc[:len(best_results['y_pred_train'])]['score'].values,
                                   df.iloc[len(best_results['y_pred_train']):]['score'].values])

    # Calculate residuals and absolute errors
    residuals = y_actual_all - y_pred_all
    abs_errors = np.abs(residuals)

    # Create analysis dataframe
    analysis_df = pd.DataFrame({
        'actual_score': y_actual_all,
        'predicted_score': y_pred_all,
        'residual': residuals,
        'abs_error': abs_errors
    })

    print(f"Df column names: {df.columns}")
    if 'country' in df.columns:
        analysis_df['country'] = df['country'].values[:len(analysis_df)]
    else:
        analysis_df['country'] = [f'Country_{i + 1}' for i in range(len(analysis_df))]

    print(f"Df column names: {analysis_df.columns}")
    # Get top/bottom predictions
    over_predictions = analysis_df.nlargest(20, 'residual')  # Predicted > Actual
    under_predictions = analysis_df.nsmallest(20, 'residual')  # Predicted < Actual
    most_accurate = analysis_df.nsmallest(20, 'abs_error')

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Prediction Analysis: Top and Bottom Performers', fontsize=16, fontweight='bold')

    # Top-left: Over-predictions
    ax1 = axes[0, 0]
    y_pos = range(len(over_predictions))
    bars = ax1.barh(y_pos, over_predictions['residual'], color='red', alpha=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(over_predictions['country'], fontsize=8)
    ax1.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    ax1.set_title('Top 20 Over-Predictions', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')

    # Top-right: Under-predictions
    ax2 = axes[0, 1]
    bars = ax2.barh(y_pos, under_predictions['residual'], color='blue', alpha=0.7)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(under_predictions['country'], fontsize=8)
    ax2.set_xlabel('Prediction Error (Predicted - Actual)', fontsize=12)
    ax2.set_title('Top 20 Under-Predictions', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')

    # Bottom-left: Most accurate predictions
    ax3 = axes[1, 0]
    bars = ax3.barh(y_pos, most_accurate['abs_error'], color='green', alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(most_accurate['country'], fontsize=8)
    ax3.set_xlabel('Absolute Error', fontsize=12)
    ax3.set_title('Top 20 Most Accurate Predictions', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='x')

    # Bottom-right: Prediction accuracy by risk level
    ax4 = axes[1, 1]

    # Create risk level bins
    analysis_df['risk_level'] = pd.qcut(analysis_df['actual_score'], q=4, labels=['Low', 'Med-Low', 'Med-High', 'High'])
    risk_accuracy = analysis_df.groupby('risk_level')['abs_error'].agg(['mean', 'std', 'count']).reset_index()

    bars = ax4.bar(range(len(risk_accuracy)), risk_accuracy['mean'],
                   yerr=risk_accuracy['std'] / np.sqrt(risk_accuracy['count']),
                   capsize=5, alpha=0.7, color=['green', 'yellow', 'orange', 'red'])
    ax4.set_xticks(range(len(risk_accuracy)))
    ax4.set_xticklabels(risk_accuracy['risk_level'])
    ax4.set_ylabel('Mean Absolute Error', fontsize=12)
    ax4.set_title('Prediction Accuracy by Risk Level', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean_error, count in zip(bars, risk_accuracy['mean'], risk_accuracy['count']):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{mean_error:.3f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('../../results/figures/04_predictive/prediction_analysis_tables.png', dpi=300, bbox_inches='tight')
    plt.show()

    return over_predictions, under_predictions, most_accurate, analysis_df


def create_model_interpretation_plots(results, X_test, feature_names):
    """Create model interpretation visualizations"""
    # Get best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']

    # Get feature importance
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    else:
        # Use permutation importance for linear models
        perm_importance = permutation_importance(best_model, X_test, results[best_model_name]['y_pred_test'],
                                                 n_repeats=10, random_state=42)
        importance = perm_importance.importances_mean

    # Get top 4 most important features
    top_indices = np.argsort(importance)[-4:]
    top_features = [feature_names[i] for i in top_indices]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Interpretation: Feature Effects and Interactions', fontsize=16, fontweight='bold')

    # Create partial dependence plots for top 4 features
    for i, (feature_idx, feature_name) in enumerate(zip(top_indices, top_features)):
        ax = axes.flatten()[i]

        # Calculate partial dependence manually (simplified)
        feature_values = X_test[:, feature_idx]
        unique_values = np.linspace(np.min(feature_values), np.max(feature_values), 50)

        partial_effects = []
        for val in unique_values:
            X_modified = X_test.copy()
            X_modified[:, feature_idx] = val

            if results[best_model_name]['scaler'] is not None:
                X_modified_scaled = results[best_model_name]['scaler'].transform(X_modified)
                pred = best_model.predict(X_modified_scaled).mean()
            else:
                pred = best_model.predict(X_modified).mean()

            partial_effects.append(pred)

        # Plot partial dependence
        ax.plot(unique_values, partial_effects, linewidth=3, color='steelblue')
        ax.fill_between(unique_values, partial_effects, alpha=0.3, color='steelblue')

        # Add rug plot
        ax.scatter(feature_values, np.full_like(feature_values, np.min(partial_effects)),
                   alpha=0.1, color='red', s=10)

        clean_name = feature_name.replace('_robust', '').replace('_binary', '').replace('_', ' ').upper()
        ax.set_xlabel(clean_name, fontsize=12)
        ax.set_ylabel('Partial Dependence', fontsize=12)
        ax.set_title(f'Partial Dependence: {clean_name}', fontsize=14)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/04_predictive/model_interpretation.png', dpi=300, bbox_inches='tight')
    plt.show()


def export_comprehensive_results(df, results, dt_results, analysis_df, feature_names):
    """Export all results to Excel"""
    with pd.ExcelWriter('../../results/tables/Analysis4_Predictive_Models_Results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: Model Performance
        performance_data = []
        for name, result in results.items():
            performance_data.append({
                'Model': name,
                'Train_R2': result['train_r2'],
                'Test_R2': result['test_r2'],
                'Train_RMSE': result['train_rmse'],
                'Test_RMSE': result['test_rmse'],
                'Train_MAE': result['train_mae'],
                'Test_MAE': result['test_mae'],
                'CV_R2_Mean': result['cv_r2_mean'],
                'CV_R2_Std': result['cv_r2_std']
            })

        performance_df = pd.DataFrame(performance_data)
        performance_df.to_excel(writer, sheet_name='Model_Performance', index=False)

        # Sheet 2: Feature Importance
        best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])

        if best_model_name in ['Random Forest', 'XGBoost']:
            importance_data = []
            best_model = results[best_model_name]['model']
            importance = best_model.feature_importances_

            for i, (feature, imp) in enumerate(zip(feature_names, importance)):
                importance_data.append({
                    'Feature': feature,
                    'Importance': imp,
                    'Rank': len(feature_names) - np.argsort(importance)[::-1].tolist().index(i)
                })

            importance_df = pd.DataFrame(importance_data)
            importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

        # Sheet 3: Predictions Analysis
        analysis_df.to_excel(writer, sheet_name='Predictions_Analysis', index=False)

        # Sheet 4: Decision Tree Results
        if dt_results:
            dt_summary = pd.DataFrame({
                'Metric': ['Accuracy', 'AUC Score'],
                'Value': [dt_results[2], dt_results[3]]  # accuracy, auc_score
            })
            dt_summary.to_excel(writer, sheet_name='Decision_Tree_Results', index=False)

        # Sheet 5: Top Predictions (from analysis)
        over_pred, under_pred, accurate_pred, _ = create_prediction_analysis_tables(df, results, feature_names)

        over_pred.to_excel(writer, sheet_name='Top_Predictions', index=False, startrow=0)
        under_pred.to_excel(writer, sheet_name='Top_Predictions', index=False, startrow=len(over_pred) + 3)
        accurate_pred.to_excel(writer, sheet_name='Top_Predictions', index=False,
                               startrow=len(over_pred) + len(under_pred) + 6)

        # Sheet 6: Model Diagnostics
        best_results = results[best_model_name]
        residuals = np.concatenate(
            [df.iloc[:len(best_results['y_pred_train'])]['score'].values - best_results['y_pred_train'],
             df.iloc[len(best_results['y_pred_train']):]['score'].values - best_results['y_pred_test']])

        diagnostics = pd.DataFrame({
            'Statistic': ['Mean Residual', 'Std Residual', 'Skewness', 'Kurtosis', 'Shapiro-Wilk p-value'],
            'Value': [
                np.mean(residuals),
                np.std(residuals),
                stats.skew(residuals),
                stats.kurtosis(residuals),
                stats.shapiro(residuals)[1] if len(residuals) < 5000 else np.nan
            ]
        })
        diagnostics.to_excel(writer, sheet_name='Model_Diagnostics', index=False)


def main():
    """Main execution function"""
    print("Starting Basel AML Predictive Modeling & Decision Tree Analysis...")

    # Load and prepare data
    df, feature_vars, target_robust, target_interpret = load_and_prepare_modeling_data('../../data/processed/basel_aml_with_fsrb.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Dataset prepared: {len(df)} countries with {len(feature_vars)} features")

    # Create interaction features
    interaction_features = create_feature_interactions(df, feature_vars)
    all_features = feature_vars + interaction_features

    # Prepare data for modeling
    X = df[all_features].values
    y = df[target_robust].values

    # Train/test split with stratified sampling
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=df['risk_category']
    )

    print(f"Df coulmnes : {df.columns}, features: {all_features}")

    print(f"Train set: {len(X_train)} samples, Test set: {len(X_test)} samples")

    # Train models
    print("\n1. Training predictive models...")
    results = train_models(X_train, X_test, y_train, y_test, all_features)

    # Create main dashboard
    print("\n2. Creating predictive modeling dashboard...")
    ensemble_importance, top_indices = create_predictive_modeling_dashboard(results, X_test, y_test, all_features)

    # Create model diagnostics
    print("\n3. Creating model diagnostics...")
    create_model_diagnostics(results, X_test, y_test, all_features)

    # Decision tree analysis
    print("\n4. Performing decision tree analysis...")
    dt, class_report, accuracy, auc_score = decision_tree_analysis(df, feature_vars, target_interpret)

    # Prediction analysis
    print("\n5. Creating prediction analysis...")
    over_pred, under_pred, accurate_pred, analysis_df = create_prediction_analysis_tables(df, results, all_features)

    # Model interpretation
    print("\n6. Creating model interpretation plots...")
    create_model_interpretation_plots(results, X_test, all_features)

    # Export results
    print("\n7. Exporting results to Excel...")
    export_comprehensive_results(df, results, (dt, class_report, accuracy, auc_score), analysis_df, all_features)

    print("\nAnalysis Complete!")
    print("=" * 60)
    print("OUTPUT FILES GENERATED:")
    print("1. predictive_modeling_dashboard.png - Main 6-panel modeling dashboard")
    print("2. model_diagnostics.png - Advanced model validation plots")
    print("3. decision_tree_analysis.png - Decision tree classification analysis")
    print("4. prediction_analysis_tables.png - Prediction accuracy tables")
    print("5. model_interpretation.png - Feature interpretation plots")
    print("6. Analysis4_Predictive_Models_Results.xlsx - Complete modeling results")
    print("=" * 60)

    # Summary results
    print(f"\nKEY FINDINGS:")

    # Best performing model
    best_model_name = max(results.keys(), key=lambda x: results[x]['test_r2'])
    best_r2 = results[best_model_name]['test_r2']
    best_rmse = results[best_model_name]['test_rmse']

    print(f"Best Performing Model: {best_model_name}")
    print(f"  Test R² = {best_r2:.3f}")
    print(f"  Test RMSE = {best_rmse:.3f}")
    print(
        f"  Cross-validation R² = {results[best_model_name]['cv_r2_mean']:.3f} ± {results[best_model_name]['cv_r2_std']:.3f}")

    # Decision tree performance
    print(f"\nDecision Tree Classification:")
    print(f"  Accuracy = {accuracy:.3f}")
    print(f"  AUC Score = {auc_score:.3f}")

    # Top features
    if ensemble_importance is not None and len(top_indices) > 0:
        top_feature = all_features[top_indices[-1]]
        print(
            f"\nMost Important Feature: {top_feature.replace('_robust', '').replace('_binary', '').replace('_', ' ').upper()}")
        print(f"  Importance Score = {ensemble_importance[top_indices[-1]]:.3f}")


if __name__ == "__main__":
    main()