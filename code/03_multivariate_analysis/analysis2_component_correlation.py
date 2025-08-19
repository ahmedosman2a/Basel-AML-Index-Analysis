# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.multivariate.manova import MANOVA
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")


def load_and_prepare_robust_data(file_path):
    """Load data and prepare robust variables for analysis"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]} countries, {df.shape[1]} variables")

        # Define robust variables
        robust_vars = ['score_robust', 'mltf_robust', 'cor_robust',
                       'fintran_robust', 'pubtran_robust', 'polleg_robust']

        # Check for robust variables, if not available, use original variables
        available_robust = [var for var in robust_vars if var in df.columns]
        if len(available_robust) < 6:
            print("Warning: Not all robust variables found. Using original variables.")
            robust_vars = ['score', 'mltf', 'cor', 'fintran', 'pubtran', 'polleg']
            available_robust = [var for var in robust_vars if var in df.columns]

        # Remove rows with missing values in robust variables
        df_clean = df.dropna(subset=available_robust + ['primary_fsrb'])
        print(f"Clean dataset: {len(df_clean)} countries with complete data")

        # Create risk categories based on score
        score_var = 'score_robust' if 'score_robust' in df_clean.columns else 'score'
        percentiles = df_clean[score_var].quantile([0.25, 0.50, 0.75]).values

        def categorize_risk(score):
            if score <= percentiles[0]:
                return 'Low Risk'
            elif score <= percentiles[1]:
                return 'Medium Risk'
            elif score <= percentiles[2]:
                return 'High Risk'
            else:
                return 'Very High Risk'

        df_clean['risk_category'] = df_clean[score_var].apply(categorize_risk)

        return df_clean, available_robust

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


def correlation_analysis_dashboard(df, robust_vars):
    """Create comprehensive correlation analysis dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Risk Component Correlation Analysis Dashboard', fontsize=16, fontweight='bold')

    # Prepare data matrix
    X = df[robust_vars].values
    var_names = [var.replace('_robust', '').upper() for var in robust_vars]

    # Calculate correlation matrix
    corr_matrix = np.corrcoef(X.T)

    # Top-left: Correlation matrix heatmap
    ax1 = axes[0, 0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    im1 = ax1.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

    # Add correlation values
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if not mask[i, j]:
                ax1.text(j, i, f'{corr_matrix[i, j]:.3f}',
                         ha='center', va='center', fontsize=9,
                         color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')

    ax1.set_xticks(range(len(var_names)))
    ax1.set_yticks(range(len(var_names)))
    ax1.set_xticklabels(var_names, rotation=45)
    ax1.set_yticklabels(var_names)
    ax1.set_title('Correlation Matrix', fontsize=14)

    # Top-right: Hierarchical clustering dendrogram
    ax2 = axes[0, 1]
    # Calculate distance matrix (1 - correlation)
    distance_matrix = 1 - np.abs(corr_matrix)
    linkage_matrix = linkage(distance_matrix, method='ward')
    dendrogram(linkage_matrix, labels=var_names, ax=ax2, orientation='top')
    ax2.set_title('Hierarchical Clustering of Variables', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)

    # Bottom-left: Partial correlation matrix (controlling for region)
    ax3 = axes[1, 0]
    # Calculate partial correlations
    from scipy.stats import pearsonr
    partial_corr = np.zeros_like(corr_matrix)

    # For each pair of variables, calculate partial correlation controlling for region
    region_dummies = pd.get_dummies(df['primary_fsrb'])
    for i in range(len(robust_vars)):
        for j in range(len(robust_vars)):
            if i != j:
                # Residualize both variables against region
                y1 = df[robust_vars[i]]
                y2 = df[robust_vars[j]]

                # Simple partial correlation (residuals method)
                from sklearn.linear_model import LinearRegression
                reg1 = LinearRegression().fit(region_dummies, y1)
                reg2 = LinearRegression().fit(region_dummies, y2)

                resid1 = y1 - reg1.predict(region_dummies)
                resid2 = y2 - reg2.predict(region_dummies)

                partial_corr[i, j], _ = pearsonr(resid1, resid2)
            else:
                partial_corr[i, j] = 1.0

    im3 = ax3.imshow(partial_corr, cmap='RdBu_r', vmin=-1, vmax=1)
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            ax3.text(j, i, f'{partial_corr[i, j]:.3f}',
                     ha='center', va='center', fontsize=9,
                     color='white' if abs(partial_corr[i, j]) > 0.5 else 'black')

    ax3.set_xticks(range(len(var_names)))
    ax3.set_yticks(range(len(var_names)))
    ax3.set_xticklabels(var_names, rotation=45)
    ax3.set_yticklabels(var_names)
    ax3.set_title('Partial Correlations (Region Controlled)', fontsize=14)

    # Bottom-right: Correlation significance matrix
    ax4 = axes[1, 1]
    p_matrix = np.zeros_like(corr_matrix)

    for i in range(len(robust_vars)):
        for j in range(len(robust_vars)):
            if i != j:
                _, p_val = pearsonr(df[robust_vars[i]], df[robust_vars[j]])
                p_matrix[i, j] = p_val
            else:
                p_matrix[i, j] = 0

    # Create significance categories
    sig_matrix = np.where(p_matrix < 0.001, 3,
                          np.where(p_matrix < 0.01, 2,
                                   np.where(p_matrix < 0.05, 1, 0)))

    im4 = ax4.imshow(sig_matrix, cmap='Reds', vmin=0, vmax=3)

    # Add significance symbols
    for i in range(len(var_names)):
        for j in range(len(var_names)):
            if i != j:
                if p_matrix[i, j] < 0.001:
                    symbol = '***'
                elif p_matrix[i, j] < 0.01:
                    symbol = '**'
                elif p_matrix[i, j] < 0.05:
                    symbol = '*'
                else:
                    symbol = 'ns'
                ax4.text(j, i, symbol, ha='center', va='center', fontsize=10, fontweight='bold')

    ax4.set_xticks(range(len(var_names)))
    ax4.set_yticks(range(len(var_names)))
    ax4.set_xticklabels(var_names, rotation=45)
    ax4.set_yticklabels(var_names)
    ax4.set_title('Correlation Significance\n(***p<0.001, **p<0.01, *p<0.05)', fontsize=14)

    # Add colorbars
    fig.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04, label='Correlation')
    fig.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, label='Partial Correlation')

    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/correlation_analysis_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    return corr_matrix, partial_corr, p_matrix


def pca_analysis_suite(df, robust_vars):
    """Comprehensive PCA analysis with multiple visualizations"""
    # Prepare data
    X = df[robust_vars].values
    var_names = [var.replace('_robust', '').upper() for var in robust_vars]

    # Standardize data (though robust variables should already be normalized)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Perform PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    # KMO and Bartlett's tests
    from factor_analyzer.factor_analyzer import calculate_kmo, calculate_bartlett_sphericity
    kmo_all, kmo_model = calculate_kmo(X_scaled)
    chi_square_value, p_value_bartlett = calculate_bartlett_sphericity(X_scaled)

    print(f"KMO Test: {kmo_model:.3f} ({'Adequate' if kmo_model > 0.7 else 'Inadequate'})")
    print(f"Bartlett's Test: χ²={chi_square_value:.2f}, p={p_value_bartlett:.6f}")

    # Plot 1: Scree Plot & Explained Variance
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig1.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')

    # Scree plot
    explained_variance = pca.explained_variance_ratio_ * 100
    cumulative_variance = np.cumsum(explained_variance)

    ax1.bar(range(1, len(explained_variance) + 1), explained_variance,
            alpha=0.7, color='steelblue', label='Individual')
    ax1.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
             'ro-', linewidth=2, markersize=8, label='Cumulative')
    ax1.axhline(y=100 / len(explained_variance), color='red', linestyle='--',
                label=f'Average ({100 / len(explained_variance):.1f}%)')
    ax1.set_xlabel('Principal Component', fontsize=12)
    ax1.set_ylabel('Explained Variance (%)', fontsize=12)
    ax1.set_title('Scree Plot', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Eigenvalues
    eigenvalues = pca.explained_variance_
    ax2.bar(range(1, len(eigenvalues) + 1), eigenvalues, alpha=0.7, color='darkgreen')
    ax2.axhline(y=1, color='red', linestyle='--', label='Kaiser Criterion (λ=1)')
    ax2.set_xlabel('Principal Component', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Eigenvalues', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/pca_explained_variance.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 2: PCA Biplot
    fig2, ax = plt.subplots(figsize=(12, 10))

    # Plot country scores
    risk_colors = {'Low Risk': '#2E8B57', 'Medium Risk': '#FFD700',
                   'High Risk': '#FF8C00', 'Very High Risk': '#DC143C'}

    for risk_cat, color in risk_colors.items():
        mask = df['risk_category'] == risk_cat
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                   c=color, alpha=0.6, s=50, label=risk_cat, edgecolors='black', linewidth=0.5)

    # Plot variable loadings as arrows
    loadings = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])

    for i, (var_name, loading) in enumerate(zip(var_names, loadings)):
        ax.arrow(0, 0, loading[0] * 3, loading[1] * 3,
                 head_width=0.1, head_length=0.1, fc='red', ec='red', alpha=0.8)
        ax.text(loading[0] * 3.5, loading[1] * 3.5, var_name,
                fontsize=12, fontweight='bold', ha='center', va='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    ax.set_xlabel(f'PC1 ({explained_variance[0]:.1f}% variance)', fontsize=14)
    ax.set_ylabel(f'PC2 ({explained_variance[1]:.1f}% variance)', fontsize=14)
    ax.set_title('PCA Biplot: Countries and Variable Loadings', fontsize=16, fontweight='bold')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/pca_biplot.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot 3: Component Loading Matrix
    fig3, ax = plt.subplots(figsize=(10, 8))

    # Show loadings for first 4 components
    loadings_matrix = pca.components_[:4].T
    im = ax.imshow(loadings_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    # Add loading values
    for i in range(len(var_names)):
        for j in range(4):
            ax.text(j, i, f'{loadings_matrix[i, j]:.3f}',
                    ha='center', va='center', fontsize=10,
                    color='white' if abs(loadings_matrix[i, j]) > 0.5 else 'black')

    ax.set_xticks(range(4))
    ax.set_yticks(range(len(var_names)))
    ax.set_xticklabels([f'PC{i + 1}\n({explained_variance[i]:.1f}%)' for i in range(4)])
    ax.set_yticklabels(var_names)
    ax.set_title('Principal Component Loadings Matrix', fontsize=16, fontweight='bold')

    plt.colorbar(im, ax=ax, label='Loading Value')
    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/pca_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    return pca, X_pca, loadings_matrix, kmo_model, chi_square_value, p_value_bartlett


def feature_importance_analysis(df, robust_vars):
    """Random Forest feature importance analysis"""
    # Prepare data
    feature_vars = [var for var in robust_vars if 'score' not in var]  # Remove target variable
    target_var = 'score_robust' if 'score_robust' in robust_vars else 'score'

    X = df[feature_vars].values
    y = df[target_var].values

    # Random Forest with cross-validation
    rf = RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)

    # Feature importance
    rf.fit(X, y)
    importance_rf = rf.feature_importances_

    # Permutation importance
    perm_importance = permutation_importance(rf, X, y, n_repeats=100, random_state=42, n_jobs=-1)
    importance_perm = perm_importance.importances_mean
    std_perm = perm_importance.importances_std

    # Cross-validation stability
    cv_scores = cross_val_score(rf, X, y, cv=10, scoring='r2')

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Random Forest Feature Importance Analysis', fontsize=16, fontweight='bold')

    feature_names = [var.replace('_robust', '').upper() for var in feature_vars]

    # Feature importance
    ax1 = axes[0, 0]
    sorted_idx = np.argsort(importance_rf)
    ax1.barh(range(len(feature_names)), importance_rf[sorted_idx], color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx])
    ax1.set_xlabel('Feature Importance', fontsize=12)
    ax1.set_title('Random Forest Feature Importance', fontsize=14)
    ax1.grid(True, alpha=0.3, axis='x')

    # Permutation importance
    ax2 = axes[0, 1]
    sorted_idx_perm = np.argsort(importance_perm)
    ax2.barh(range(len(feature_names)), importance_perm[sorted_idx_perm],
             xerr=std_perm[sorted_idx_perm], color='darkgreen', alpha=0.7, capsize=5)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx_perm])
    ax2.set_xlabel('Permutation Importance', fontsize=12)
    ax2.set_title('Permutation Feature Importance', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='x')

    # Comparison plot
    ax3 = axes[1, 0]
    ax3.scatter(importance_rf, importance_perm, alpha=0.7, s=100, color='red')
    for i, name in enumerate(feature_names):
        ax3.annotate(name, (importance_rf[i], importance_perm[i]),
                     xytext=(5, 5), textcoords='offset points', fontsize=10)

    # Add diagonal line
    max_imp = max(max(importance_rf), max(importance_perm))
    ax3.plot([0, max_imp], [0, max_imp], 'k--', alpha=0.5)
    ax3.set_xlabel('Random Forest Importance', fontsize=12)
    ax3.set_ylabel('Permutation Importance', fontsize=12)
    ax3.set_title('Importance Method Comparison', fontsize=14)
    ax3.grid(True, alpha=0.3)

    # Cross-validation scores
    ax4 = axes[1, 1]
    ax4.boxplot(cv_scores)
    ax4.set_ylabel('R² Score', fontsize=12)
    ax4.set_title(f'Cross-Validation Stability\nMean R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    return importance_rf, importance_perm, std_perm, cv_scores


def regional_component_analysis(df, robust_vars):
    """Regional analysis of risk components"""
    # Filter regions with sufficient sample size
    region_counts = df['primary_fsrb'].value_counts()
    valid_regions = region_counts[region_counts >= 3].index
    df_regional = df[df['primary_fsrb'].isin(valid_regions)].copy()

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Regional Risk Component Analysis', fontsize=16, fontweight='bold')

    # Violin plots for each component
    ax1 = axes[0, 0]
    component_vars = [var for var in robust_vars if 'score' not in var]

    # Prepare data for violin plot
    plot_data = []
    for region in valid_regions:
        for var in component_vars:
            values = df_regional[df_regional['primary_fsrb'] == region][var].values
            for val in values:
                plot_data.append({'Region': region, 'Component': var.replace('_robust', '').upper(), 'Value': val})

    plot_df = pd.DataFrame(plot_data)
    sns.violinplot(data=plot_df, x='Component', y='Value', hue='Region', ax=ax1)
    ax1.set_title('Component Distributions by Region', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    # Regional component heatmap
    ax2 = axes[0, 1]
    regional_means = df_regional.groupby('primary_fsrb')[component_vars].mean()
    regional_means.columns = [col.replace('_robust', '').upper() for col in regional_means.columns]

    im = sns.heatmap(regional_means, annot=True, fmt='.2f', cmap='RdYlBu_r',
                     ax=ax2, cbar_kws={'label': 'Mean Component Value'})
    ax2.set_title('Regional Component Means Heatmap', fontsize=14)
    ax2.set_ylabel('Region (FSRB)', fontsize=12)

    # Radar plots for top 6 regions (by sample size)
    top_regions = region_counts.head(6).index
    ax3 = axes[1, 0]
    ax4 = axes[1, 1]

    # Prepare radar plot data
    component_names = [var.replace('_robust', '').upper() for var in component_vars]

    # Normalize data to 0-1 scale for radar plots
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_regional[component_vars])
    df_normalized = pd.DataFrame(normalized_data, columns=component_vars, index=df_regional.index)
    df_normalized['primary_fsrb'] = df_regional['primary_fsrb'].values

    # Create radar plot function
    def create_radar_plot(ax, region_data, title):
        angles = np.linspace(0, 2 * np.pi, len(component_names), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        colors = plt.cm.Set3(np.linspace(0, 1, len(region_data)))

        for i, (region, values) in enumerate(region_data.items()):
            values_list = values.tolist()
            values_list += values_list[:1]  # Complete the circle

            ax.plot(angles, values_list, 'o-', linewidth=2, label=region, color=colors[i])
            ax.fill(angles, values_list, alpha=0.25, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(component_names)
        ax.set_ylim(-2, 2)
        ax.set_title(title, fontsize=14)
        ax.legend(bbox_to_anchor=(1.1, 1.1))
        ax.grid(True)

    # First radar plot (top 3 regions)
    top3_regions = top_regions[:3]
    region_means_top3 = {}
    for region in top3_regions:
        region_means_top3[region] = df_normalized[df_normalized['primary_fsrb'] == region][component_vars].mean()

    create_radar_plot(ax3, region_means_top3, 'Top 3 Regions by Sample Size')

    # Second radar plot (next 3 regions)
    if len(top_regions) >= 6:
        next3_regions = top_regions[3:6]
        region_means_next3 = {}
        for region in next3_regions:
            region_means_next3[region] = df_normalized[df_normalized['primary_fsrb'] == region][component_vars].mean()

        create_radar_plot(ax4, region_means_next3, 'Next 3 Regions by Sample Size')
    else:
        ax4.text(0.5, 0.5, 'Insufficient regions\nfor second radar plot',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=14)
        ax4.set_title('Regional Radar Plot 2', fontsize=14)

    plt.tight_layout()
    plt.savefig('../../results/figures/02_multivariate/regional_component_profiles.png', dpi=300, bbox_inches='tight')
    plt.show()

    # MANOVA analysis
    try:
        # Prepare data for MANOVA
        manova_formula = ' + '.join(component_vars) + ' ~ C(primary_fsrb)'
        manova = MANOVA.from_formula(manova_formula, data=df_regional)
        manova_results = manova.mv_test()

        print("\nMANOVA Results:")
        print(manova_results)

    except Exception as e:
        print(f"MANOVA analysis failed: {str(e)}")
        manova_results = None

    return regional_means, manova_results


def advanced_statistical_tests(df, robust_vars):
    """Perform advanced statistical tests"""
    X = df[robust_vars].values

    # Box's M test for homogeneity of covariance matrices
    def boxes_m_test(data, groups):
        """Simplified Box's M test implementation"""
        try:
            from sklearn.covariance import LedoitWolf
            unique_groups = np.unique(groups)
            n_groups = len(unique_groups)
            n_vars = data.shape[1]

            # Calculate pooled covariance matrix
            pooled_cov = LedoitWolf().fit(data).covariance_

            # Calculate group covariance matrices
            group_covs = []
            group_sizes = []

            for group in unique_groups:
                group_data = data[groups == group]
                if len(group_data) > n_vars:  # Ensure we have enough samples
                    group_cov = LedoitWolf().fit(group_data).covariance_
                    group_covs.append(group_cov)
                    group_sizes.append(len(group_data))

            # Box's M statistic (simplified)
            m_stat = 0
            for i, (cov, n) in enumerate(zip(group_covs, group_sizes)):
                m_stat += (n - 1) * (np.log(np.linalg.det(pooled_cov)) - np.log(np.linalg.det(cov)))

            return m_stat, "Box's M test completed"

        except Exception as e:
            return np.nan, f"Box's M test failed: {str(e)}"

    # Perform Box's M test
    regions = df['primary_fsrb'].values
    box_m_stat, box_m_msg = boxes_m_test(X, regions)

    print(f"Box's M Test: {box_m_msg}")
    if not np.isnan(box_m_stat):
        print(f"Box's M Statistic: {box_m_stat:.3f}")

    # Discriminant analysis for regional classification
    try:
        # Filter regions with sufficient samples
        region_counts = pd.Series(regions).value_counts()
        valid_regions = region_counts[region_counts >= 3].index

        valid_mask = np.isin(regions, valid_regions)
        X_valid = X[valid_mask]
        regions_valid = regions[valid_mask]

        # Linear Discriminant Analysis
        lda = LinearDiscriminantAnalysis()
        lda.fit(X_valid, regions_valid)

        # Cross-validation accuracy
        cv_scores = cross_val_score(lda, X_valid, regions_valid, cv=5, scoring='accuracy')
        lda_accuracy = cv_scores.mean()

        print(f"LDA Classification Accuracy: {lda_accuracy:.3f} ± {cv_scores.std():.3f}")

    except Exception as e:
        print(f"LDA analysis failed: {str(e)}")
        lda_accuracy = np.nan
        cv_scores = np.array([])

    return box_m_stat, lda_accuracy, cv_scores


def export_comprehensive_results(df, robust_vars, corr_matrix, partial_corr, p_matrix,
                                 pca, importance_rf, importance_perm, regional_means, manova_results):
    """Export all results to Excel"""
    with pd.ExcelWriter('../../results/tables/Analysis2_Component_Analysis_Results.xlsx', engine='openpyxl') as writer:
        # Sheet 1: Correlation Matrix
        var_names = [var.replace('_robust', '').upper() for var in robust_vars]
        corr_df = pd.DataFrame(corr_matrix, index=var_names, columns=var_names)
        corr_df.to_excel(writer, sheet_name='Correlation_Matrix')

        # Add partial correlations
        partial_df = pd.DataFrame(partial_corr, index=var_names, columns=var_names)
        partial_df.to_excel(writer, sheet_name='Correlation_Matrix', startrow=len(corr_df) + 3)

        # Sheet 2: PCA Results
        # Explained variance
        pca_summary = pd.DataFrame({
            'Component': [f'PC{i + 1}' for i in range(len(pca.explained_variance_))],
            'Eigenvalue': pca.explained_variance_,
            'Explained_Variance_Ratio': pca.explained_variance_ratio_,
            'Cumulative_Variance': np.cumsum(pca.explained_variance_ratio_)
        })
        pca_summary.to_excel(writer, sheet_name='PCA_Results', index=False)

        # Component loadings
        component_names = [var.replace('_robust', '').upper() for var in robust_vars]
        loadings_df = pd.DataFrame(
            pca.components_[:4].T,
            index=component_names,
            columns=[f'PC{i + 1}' for i in range(4)]
        )
        loadings_df.to_excel(writer, sheet_name='PCA_Results', startrow=len(pca_summary) + 3)

        # Sheet 3: Feature Importance
        feature_names = [var.replace('_robust', '').upper() for var in robust_vars if 'score' not in var]
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Random_Forest_Importance': importance_rf,
            'Permutation_Importance': importance_perm
        })
        importance_df.to_excel(writer, sheet_name='Feature_Importance', index=False)

        # Sheet 4: Regional Statistics
        if regional_means is not None:
            regional_means.to_excel(writer, sheet_name='Regional_Statistics')

        # Sheet 5: Statistical Tests
        test_results = pd.DataFrame({
            'Test': ['Correlation Significance', 'MANOVA', 'Regional Classification'],
            'Details': ['See correlation p-values',
                        'See MANOVA results' if manova_results is not None else 'Failed',
                        'See LDA accuracy']
        })
        test_results.to_excel(writer, sheet_name='Statistical_Tests', index=False)


def main():
    """Main execution function"""
    print("Starting Basel AML Component Correlation & Dimensionality Analysis...")

    # Load data
    df, robust_vars = load_and_prepare_robust_data('../../data/processed/basel_aml_with_fsrb.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Analysis will use {len(robust_vars)} robust variables: {robust_vars}")

    # Correlation analysis
    print("\n1. Performing correlation analysis...")
    corr_matrix, partial_corr, p_matrix = correlation_analysis_dashboard(df, robust_vars)

    # PCA analysis
    print("\n2. Performing PCA analysis...")
    pca, X_pca, loadings_matrix, kmo_score, bartlett_chi2, bartlett_p = pca_analysis_suite(df, robust_vars)

    # Feature importance analysis
    print("\n3. Performing feature importance analysis...")
    importance_rf, importance_perm, std_perm, cv_scores = feature_importance_analysis(df, robust_vars)

    # Regional analysis
    print("\n4. Performing regional component analysis...")
    regional_means, manova_results = regional_component_analysis(df, robust_vars)

    # Advanced statistical tests
    print("\n5. Performing advanced statistical tests...")
    box_m_stat, lda_accuracy, lda_cv_scores = advanced_statistical_tests(df, robust_vars)

    # Export results
    print("\n6. Exporting results to Excel...")
    export_comprehensive_results(df, robust_vars, corr_matrix, partial_corr, p_matrix,
                                 pca, importance_rf, importance_perm, regional_means, manova_results)

    print("\nAnalysis Complete!")
    print("=" * 60)
    print("OUTPUT FILES GENERATED:")
    print("1. correlation_analysis_dashboard.png - Correlation analysis suite")
    print("2. pca_explained_variance.png - PCA scree plot and variance")
    print("3. pca_biplot.png - PCA biplot with countries and loadings")
    print("4. pca_loadings_heatmap.png - Component loadings matrix")
    print("5. feature_importance_analysis.png - Random Forest importance")
    print("6. regional_component_profiles.png - Regional analysis suite")
    print("7. Analysis2_Component_Analysis_Results.xlsx - Complete results")
    print("=" * 60)

    # Summary results
    print(f"\nKEY FINDINGS:")
    print(f"KMO Sampling Adequacy: {kmo_score:.3f}")
    print(f"Bartlett's Sphericity: χ²={bartlett_chi2:.2f}, p={bartlett_p:.6f}")
    print(f"PC1 Explained Variance: {pca.explained_variance_ratio_[0] * 100:.1f}%")
    print(f"PC2 Explained Variance: {pca.explained_variance_ratio_[1] * 100:.1f}%")
    print(f"Cumulative Variance (PC1+PC2): {sum(pca.explained_variance_ratio_[:2]) * 100:.1f}%")
    if not np.isnan(lda_accuracy):
        print(f"Regional Classification Accuracy: {lda_accuracy:.3f}")

    feature_names = [var.replace('_robust', '').upper() for var in robust_vars if 'score' not in var]
    most_important = feature_names[np.argmax(importance_rf)]
    print(f"Most Important Feature (RF): {most_important}")


if __name__ == "__main__":
    main()