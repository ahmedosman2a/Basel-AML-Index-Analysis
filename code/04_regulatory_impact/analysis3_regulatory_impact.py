# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, mannwhitneyu, ttest_ind
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import itertools
import warnings

warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("Set2")


def load_and_prepare_sanctions_data(file_path):
    """Load data and prepare sanction variables"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded: {df.shape[0]} countries, {df.shape[1]} variables")

        # Define sanction binary variables
        sanction_vars = [
            'ofac_binary', 'un_sc_binary', 'eu_restrictive_measures_binary',
            'fatf_grey_binary', 'fatf_black_binary', 'aus_sanctions_binary',
            'uk_sanctions_binary', 'can_sanctions_binary', 'jp_sanctions_binary',
            'ch_sanctions_binary', 'other_sanctions_binary'
        ]

        # Check available sanction variables
        available_sanctions = [var for var in sanction_vars if var in df.columns]
        missing_sanctions = [var for var in sanction_vars if var not in df.columns]

        if missing_sanctions:
            print(f"Warning: Missing sanction variables: {missing_sanctions}")

        print(f"Available sanction variables: {len(available_sanctions)}")

        # Check for composite measures
        composite_vars = ['sanction_total', 'sanction_weighted', 'sanction_level']
        available_composite = [var for var in composite_vars if var in df.columns]

        if not available_composite:
            print("Creating sanction_total from available binary variables...")
            df['sanction_total'] = df[available_sanctions].sum(axis=1)

        # Clean data - remove rows with missing key variables
        key_vars = ['score', 'primary_fsrb'] + available_sanctions
        if 'score_robust' in df.columns:
            key_vars.append('score_robust')

        df_clean = df.dropna(subset=key_vars)
        print(f"Clean dataset: {len(df_clean)} countries with complete sanction data")

        return df_clean, available_sanctions

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None, None


def calculate_cohens_d(group1, group2):
    """Calculate Cohen's d effect size with confidence interval"""
    n1, n2 = len(group1), len(group2)

    # Calculate means and standard deviations
    m1, m2 = np.mean(group1), np.mean(group2)
    s1, s2 = np.std(group1, ddof=1), np.std(group2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))

    # Cohen's d
    d = (m1 - m2) / pooled_std

    # 95% confidence interval (approximate)
    se_d = np.sqrt((n1 + n2) / (n1 * n2) + d ** 2 / (2 * (n1 + n2)))
    ci_lower = d - 1.96 * se_d
    ci_upper = d + 1.96 * se_d

    return d, ci_lower, ci_upper


def individual_regulatory_impact_analysis(df, sanction_vars):
    """Analyze individual regulatory impact with violin plots"""
    # Set up the subplot grid (3x4 for 11 sanctions + 1 summary)
    fig, axes = plt.subplots(3, 4, figsize=(20, 16))
    fig.suptitle('Individual Regulatory Impact on AML Risk Scores', fontsize=16, fontweight='bold')

    axes = axes.flatten()

    # Colors for sanctioned vs not sanctioned
    colors = ['#33A02C', '#E31A1C']  # Green for not sanctioned, Red for sanctioned

    effect_sizes = {}
    statistical_results = {}

    for i, sanction in enumerate(sanction_vars):
        if i >= len(axes):
            break

        ax = axes[i]

        # Split data by sanction status
        not_sanctioned = df[df[sanction] == 0]['score'].values
        sanctioned = df[df[sanction] == 1]['score'].values

        if len(sanctioned) < 3 or len(not_sanctioned) < 3:
            ax.text(0.5, 0.5, f'Insufficient data\nfor {sanction}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(sanction.replace('_binary', '').upper(), fontsize=12)
            continue

        # Create violin plot
        data_for_plot = [not_sanctioned, sanctioned]
        parts = ax.violinplot(data_for_plot, positions=[0, 1], showmeans=True, showextrema=True)

        # Color the violin plots
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

        # Statistical tests
        # Test for normality
        _, p_norm_0 = stats.shapiro(not_sanctioned) if len(not_sanctioned) < 5000 else (None, 0.001)
        _, p_norm_1 = stats.shapiro(sanctioned) if len(sanctioned) < 5000 else (None, 0.001)

        # Choose appropriate test
        if p_norm_0 > 0.05 and p_norm_1 > 0.05:
            # Use t-test if both groups are normal
            t_stat, p_value = ttest_ind(sanctioned, not_sanctioned)
            test_used = "t-test"
        else:
            # Use Mann-Whitney U test if non-normal
            u_stat, p_value = mannwhitneyu(sanctioned, not_sanctioned, alternative='two-sided')
            test_used = "Mann-Whitney U"

        # Calculate Cohen's d
        d, d_ci_lower, d_ci_upper = calculate_cohens_d(sanctioned, not_sanctioned)

        # Effect size interpretation
        if abs(d) < 0.2:
            effect_interp = "Negligible"
        elif abs(d) < 0.5:
            effect_interp = "Small"
        elif abs(d) < 0.8:
            effect_interp = "Medium"
        else:
            effect_interp = "Large"

        # Store results
        effect_sizes[sanction] = d
        statistical_results[sanction] = {
            'cohens_d': d,
            'cohens_d_ci': (d_ci_lower, d_ci_upper),
            'p_value': p_value,
            'test_used': test_used,
            'effect_interpretation': effect_interp,
            'n_sanctioned': len(sanctioned),
            'n_not_sanctioned': len(not_sanctioned),
            'mean_sanctioned': np.mean(sanctioned),
            'mean_not_sanctioned': np.mean(not_sanctioned)
        }

        # Add statistical annotations
        ax.text(0.5, 0.95, f"Cohen's d = {d:.3f} ({effect_interp})",
                transform=ax.transAxes, ha='center', fontsize=10, fontweight='bold')

        # Format p-value
        if p_value < 0.001:
            p_str = "p < 0.001***"
        elif p_value < 0.01:
            p_str = f"p = {p_value:.3f}**"
        elif p_value < 0.05:
            p_str = f"p = {p_value:.3f}*"
        else:
            p_str = f"p = {p_value:.3f}"

        ax.text(0.5, 0.85, p_str, transform=ax.transAxes, ha='center', fontsize=10)
        ax.text(0.5, 0.75, f"n₀={len(not_sanctioned)}, n₁={len(sanctioned)}",
                transform=ax.transAxes, ha='center', fontsize=9)

        # Formatting
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Not Sanctioned', 'Sanctioned'])
        ax.set_ylabel('Risk Score', fontsize=10)
        ax.set_title(sanction.replace('_binary', '').replace('_', ' ').upper(), fontsize=12)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for i in range(len(sanction_vars), len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/individual_regulatory_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Create effect size ranking chart
    fig2, ax = plt.subplots(figsize=(12, 8))

    # Sort by absolute effect size
    sorted_sanctions = sorted(effect_sizes.items(), key=lambda x: abs(x[1]), reverse=True)
    sanction_names = [s[0].replace('_binary', '').replace('_', ' ').upper() for s, d in sorted_sanctions]
    effect_values = [d for s, d in sorted_sanctions]

    # Color by effect size magnitude
    colors_effect = []
    for d in effect_values:
        abs_d = abs(d)
        if abs_d >= 0.8:
            colors_effect.append('#D62728')  # Red for large
        elif abs_d >= 0.5:
            colors_effect.append('#FF7F0E')  # Orange for medium
        elif abs_d >= 0.2:
            colors_effect.append('#2CA02C')  # Green for small
        else:
            colors_effect.append('#1F77B4')  # Blue for negligible

    bars = ax.barh(range(len(sanction_names)), effect_values, color=colors_effect, alpha=0.8)
    ax.set_yticks(range(len(sanction_names)))
    ax.set_yticklabels(sanction_names)
    ax.set_xlabel("Cohen's d (Effect Size)", fontsize=12)
    ax.set_title("Regulatory Impact Effect Size Ranking", fontsize=14, fontweight='bold')

    # Add effect size reference lines
    ax.axvline(x=0.2, color='gray', linestyle='--', alpha=0.5, label='Small effect')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Medium effect')
    ax.axvline(x=0.8, color='gray', linestyle='--', alpha=0.5, label='Large effect')
    ax.axvline(x=-0.2, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=-0.8, color='gray', linestyle='--', alpha=0.5)

    # Add value labels
    for bar, value in zip(bars, effect_values):
        width = bar.get_width()
        ax.text(width + (0.05 if width >= 0 else -0.05), bar.get_y() + bar.get_height() / 2,
                f'{value:.3f}', ha='left' if width >= 0 else 'right', va='center', fontsize=9)

    ax.grid(True, alpha=0.3, axis='x')
    ax.legend()

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/effect_size_ranking.png', dpi=300, bbox_inches='tight')
    plt.show()

    return statistical_results


def fatf_regional_analysis(df):
    """Analyze FATF impact by region"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('FATF Regional Impact Analysis', fontsize=16, fontweight='bold')

    # Filter regions with sufficient sample size
    region_counts = df['primary_fsrb'].value_counts()
    valid_regions = region_counts[region_counts >= 5].index
    df_regional = df[df['primary_fsrb'].isin(valid_regions)].copy()

    # Top-left: FATF Grey List impact by region
    ax1 = axes[0, 0]
    if 'fatf_grey_binary' in df.columns:
        sns.boxplot(data=df_regional, x='primary_fsrb', y='score', hue='fatf_grey_binary', ax=ax1)
        ax1.set_title('FATF Grey List Impact by Region', fontsize=14)
        ax1.set_xlabel('Region (FSRB)', fontsize=12)
        ax1.set_ylabel('Risk Score', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        ax1.legend(title='FATF Grey List', labels=['No', 'Yes'])
    else:
        ax1.text(0.5, 0.5, 'FATF Grey List data\nnot available',
                 ha='center', va='center', transform=ax1.transAxes, fontsize=12)
        ax1.set_title('FATF Grey List Impact by Region', fontsize=14)

    # Top-right: FATF Black List impact by region
    ax2 = axes[0, 1]
    if 'fatf_black_binary' in df.columns:
        sns.boxplot(data=df_regional, x='primary_fsrb', y='score', hue='fatf_black_binary', ax=ax2)
        ax2.set_title('FATF Black List Impact by Region', fontsize=14)
        ax2.set_xlabel('Region (FSRB)', fontsize=12)
        ax2.set_ylabel('Risk Score', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        ax2.legend(title='FATF Black List', labels=['No', 'Yes'])
    else:
        ax2.text(0.5, 0.5, 'FATF Black List data\nnot available',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=12)
        ax2.set_title('FATF Black List Impact by Region', fontsize=14)

    # Bottom-left: FATF compliance rates by region
    ax3 = axes[1, 0]
    if 'fatf_grey_binary' in df.columns and 'fatf_black_binary' in df.columns:
        # Calculate compliance rates (not on any FATF list)
        df_regional['fatf_compliant'] = ((df_regional['fatf_grey_binary'] == 0) &
                                         (df_regional['fatf_black_binary'] == 0)).astype(int)

        compliance_rates = df_regional.groupby('primary_fsrb')['fatf_compliant'].mean()
        bars = ax3.bar(range(len(compliance_rates)), compliance_rates.values,
                       color='steelblue', alpha=0.7)
        ax3.set_xticks(range(len(compliance_rates)))
        ax3.set_xticklabels(compliance_rates.index, rotation=45)
        ax3.set_ylabel('FATF Compliance Rate', fontsize=12)
        ax3.set_title('FATF Compliance Rates by Region', fontsize=14)
        ax3.set_ylim(0, 1)

        # Add value labels
        for bar, rate in zip(bars, compliance_rates.values):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                     f'{rate:.2f}', ha='center', va='bottom', fontsize=10)
    else:
        ax3.text(0.5, 0.5, 'FATF compliance data\nnot available',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('FATF Compliance Rates by Region', fontsize=14)

    # Bottom-right: Two-way interaction plot
    ax4 = axes[1, 1]
    if 'fatf_grey_binary' in df.columns:
        # Create interaction plot
        interaction_data = df_regional.groupby(['primary_fsrb', 'fatf_grey_binary'])['score'].mean().unstack()

        if interaction_data.shape[1] == 2:  # Both levels present
            for i, region in enumerate(interaction_data.index):
                ax4.plot([0, 1], interaction_data.loc[region], 'o-',
                         label=region, linewidth=2, markersize=6)

            ax4.set_xticks([0, 1])
            ax4.set_xticklabels(['Not Grey Listed', 'Grey Listed'])
            ax4.set_ylabel('Mean Risk Score', fontsize=12)
            ax4.set_title('FATF Grey List × Region Interaction', fontsize=14)
            ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Insufficient variation\nfor interaction plot',
                     ha='center', va='center', transform=ax4.transAxes, fontsize=12)
            ax4.set_title('FATF Grey List × Region Interaction', fontsize=14)
    else:
        ax4.text(0.5, 0.5, 'FATF interaction data\nnot available',
                 ha='center', va='center', transform=ax4.transAxes, fontsize=12)
        ax4.set_title('FATF Grey List × Region Interaction', fontsize=14)

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/fatf_regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Perform two-way ANOVA if data available
    anova_results = {}
    if 'fatf_grey_binary' in df.columns:
        try:
            formula = 'score ~ C(fatf_grey_binary) * C(primary_fsrb)'
            model = ols(formula, data=df_regional).fit()
            anova_table = anova_lm(model, typ=2)
            anova_results['fatf_grey_region'] = anova_table
            print("Two-way ANOVA (FATF Grey × Region):")
            print(anova_table)
        except Exception as e:
            print(f"ANOVA failed: {str(e)}")

    return anova_results


def cumulative_regulatory_impact(df):
    """Analyze cumulative regulatory impact"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cumulative Regulatory Impact Analysis', fontsize=16, fontweight='bold')

    # Ensure sanction_total exists
    if 'sanction_total' not in df.columns:
        sanction_vars = [col for col in df.columns if
                         '_binary' in col and col != 'fatf_grey_binary' and col != 'fatf_black_binary']
        df['sanction_total'] = df[sanction_vars].sum(axis=1)

    # Top-left: Scatter plot with regression line
    ax1 = axes[0, 0]
    x = df['sanction_total']
    y = df['score']

    ax1.scatter(x, y, alpha=0.6, s=50, color='steelblue')

    # Fit regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax1.plot(x, p(x), "r--", alpha=0.8, linewidth=2)

    # Calculate R-squared
    r_squared = r2_score(y, p(x))
    correlation, p_corr = stats.pearsonr(x, y)

    ax1.text(0.05, 0.95, f'R² = {r_squared:.3f}\nr = {correlation:.3f}\np = {p_corr:.3f}',
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax1.set_xlabel('Total Number of Sanctions', fontsize=12)
    ax1.set_ylabel('Risk Score', fontsize=12)
    ax1.set_title('Risk Score vs Sanction Count', fontsize=14)
    ax1.grid(True, alpha=0.3)

    # Top-right: Box plot by sanction count categories
    ax2 = axes[0, 1]

    # Create sanction categories
    df['sanction_category'] = pd.cut(df['sanction_total'],
                                     bins=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5, np.inf],
                                     labels=['0', '1', '2', '3', '4', '5+'])

    # Remove empty categories
    category_counts = df['sanction_category'].value_counts()
    valid_categories = category_counts[category_counts > 0].index
    df_valid = df[df['sanction_category'].isin(valid_categories)]

    sns.boxplot(data=df_valid, x='sanction_category', y='score', ax=ax2)
    ax2.set_xlabel('Number of Sanctions', fontsize=12)
    ax2.set_ylabel('Risk Score', fontsize=12)
    ax2.set_title('Risk Score Distribution by Sanction Count', fontsize=14)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Mean progression with error bars
    ax3 = axes[1, 0]

    # Calculate means and standard errors for each category
    category_stats = df_valid.groupby('sanction_category')['score'].agg(['mean', 'std', 'count']).reset_index()
    category_stats['se'] = category_stats['std'] / np.sqrt(category_stats['count'])

    bars = ax3.bar(range(len(category_stats)), category_stats['mean'],
                   yerr=category_stats['se'], capsize=5, alpha=0.7, color='darkorange')
    ax3.set_xticks(range(len(category_stats)))
    ax3.set_xticklabels(category_stats['sanction_category'])
    ax3.set_xlabel('Number of Sanctions', fontsize=12)
    ax3.set_ylabel('Mean Risk Score', fontsize=12)
    ax3.set_title('Mean Risk Score Progression', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, mean_val, count in zip(bars, category_stats['mean'], category_stats['count']):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                 f'{mean_val:.2f}\n(n={count})', ha='center', va='bottom', fontsize=9)

    # Bottom-right: Trend analysis
    ax4 = axes[1, 1]

    # Linear trend
    x_vals = category_stats.index
    y_vals = category_stats['mean']

    # Fit linear and quadratic trends
    linear_coef = np.polyfit(x_vals, y_vals, 1)
    quadratic_coef = np.polyfit(x_vals, y_vals, 2)

    linear_fit = np.poly1d(linear_coef)
    quadratic_fit = np.poly1d(quadratic_coef)

    x_smooth = np.linspace(0, len(category_stats) - 1, 100)

    ax4.scatter(x_vals, y_vals, s=100, color='red', zorder=5, label='Observed means')
    ax4.plot(x_smooth, linear_fit(x_smooth), '--', color='blue', linewidth=2, label='Linear trend')
    ax4.plot(x_smooth, quadratic_fit(x_smooth), '-', color='green', linewidth=2, label='Quadratic trend')

    ax4.set_xticks(x_vals)
    ax4.set_xticklabels(category_stats['sanction_category'])
    ax4.set_xlabel('Sanction Category', fontsize=12)
    ax4.set_ylabel('Mean Risk Score', fontsize=12)
    ax4.set_title('Trend Analysis', fontsize=14)
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/cumulative_regulatory_impact.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Statistical trend tests
    trend_results = {
        'linear_correlation': correlation,
        'linear_p_value': p_corr,
        'linear_r_squared': r_squared,
        'category_means': category_stats[['sanction_category', 'mean', 'std', 'count']].to_dict('records')
    }

    return trend_results


def regulatory_overlap_analysis(df, sanction_vars):
    """Analyze regulatory overlap patterns"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Regulatory Overlap Analysis', fontsize=16, fontweight='bold')

    # Create co-occurrence matrix
    sanction_matrix = df[sanction_vars].values
    co_occurrence = np.dot(sanction_matrix.T, sanction_matrix)

    # Top-left: Co-occurrence heatmap
    ax1 = axes[0, 0]
    sanction_labels = [s.replace('_binary', '').replace('_', ' ').upper() for s in sanction_vars]

    im = ax1.imshow(co_occurrence, cmap='YlOrRd', aspect='auto')
    ax1.set_xticks(range(len(sanction_labels)))
    ax1.set_yticks(range(len(sanction_labels)))
    ax1.set_xticklabels(sanction_labels, rotation=45, ha='right')
    ax1.set_yticklabels(sanction_labels)

    # Add co-occurrence values
    for i in range(len(sanction_labels)):
        for j in range(len(sanction_labels)):
            text = ax1.text(j, i, int(co_occurrence[i, j]),
                            ha="center", va="center",
                            color="white" if co_occurrence[i, j] > co_occurrence.max() / 2 else "black")

    ax1.set_title('Sanction Co-occurrence Matrix', fontsize=14)
    plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # Top-right: Network-style visualization (simplified as correlation plot)
    ax2 = axes[0, 1]
    # Calculate correlation matrix for sanctions
    sanction_corr = np.corrcoef(sanction_matrix.T)

    # Only show strong correlations
    threshold = 0.3
    strong_corr = np.where(np.abs(sanction_corr) >= threshold, sanction_corr, 0)

    im2 = ax2.imshow(strong_corr, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax2.set_xticks(range(len(sanction_labels)))
    ax2.set_yticks(range(len(sanction_labels)))
    ax2.set_xticklabels(sanction_labels, rotation=45, ha='right')
    ax2.set_yticklabels(sanction_labels)
    ax2.set_title(f'Strong Sanction Correlations (|r| ≥ {threshold})', fontsize=14)
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # Bottom-left: Most frequent sanction combinations
    ax3 = axes[1, 0]

    # Find all unique sanction combinations
    combinations = []
    combination_counts = []

    for idx, row in df.iterrows():
        active_sanctions = [sanction_vars[i] for i, val in enumerate(row[sanction_vars]) if val == 1]
        if len(active_sanctions) > 1:  # Only combinations with 2+ sanctions
            combo = tuple(sorted(active_sanctions))
            if combo not in combinations:
                combinations.append(combo)
                combination_counts.append(1)
            else:
                idx_combo = combinations.index(combo)
                combination_counts[idx_combo] += 1

    # Get top 15 combinations
    if combinations:
        combo_data = list(zip(combinations, combination_counts))
        combo_data.sort(key=lambda x: x[1], reverse=True)
        top_combos = combo_data[:15]

        combo_labels = [' + '.join([s.replace('_binary', '').replace('_', ' ').upper() for s in combo[0]])
                        for combo in top_combos]
        combo_counts = [combo[1] for combo in top_combos]

        bars = ax3.barh(range(len(combo_labels)), combo_counts, color='steelblue', alpha=0.7)
        ax3.set_yticks(range(len(combo_labels)))
        ax3.set_yticklabels(combo_labels, fontsize=8)
        ax3.set_xlabel('Frequency', fontsize=12)
        ax3.set_title('Top 15 Sanction Combinations', fontsize=14)

        # Add value labels
        for bar, count in zip(bars, combo_counts):
            ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                     str(count), ha='left', va='center', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No significant\nsanction combinations found',
                 ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Top 15 Sanction Combinations', fontsize=14)

    # Bottom-right: Sanction burden distribution
    ax4 = axes[1, 1]

    sanction_totals = df[sanction_vars].sum(axis=1)
    sanction_dist = sanction_totals.value_counts().sort_index()

    bars = ax4.bar(sanction_dist.index, sanction_dist.values, alpha=0.7, color='darkgreen')
    ax4.set_xlabel('Number of Sanctions', fontsize=12)
    ax4.set_ylabel('Number of Countries', fontsize=12)
    ax4.set_title('Distribution of Sanction Burden', fontsize=14)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add percentage labels
    total_countries = len(df)
    for bar, count in zip(bars, sanction_dist.values):
        percentage = (count / total_countries) * 100
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f'{count}\n({percentage:.1f}%)', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/regulatory_overlap_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    overlap_results = {
        'co_occurrence_matrix': co_occurrence,
        'top_combinations': top_combos if 'top_combos' in locals() else [],
        'sanction_distribution': sanction_dist.to_dict()
    }

    return overlap_results


def fatf_interaction_effects_analysis(df):
    """Analyze FATF interaction effects with major sanctions"""
    major_sanctions = ['ofac_binary', 'un_sc_binary', 'eu_restrictive_measures_binary']
    fatf_types = ['fatf_grey_binary', 'fatf_black_binary']

    # Filter to only include available sanctions
    available_major = [s for s in major_sanctions if s in df.columns]
    available_fatf = [f for f in fatf_types if f in df.columns]

    if not available_major or not available_fatf:
        print("Insufficient FATF or major sanction data for interaction analysis")
        return {}

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('FATF Interaction Effects with Major Sanctions', fontsize=16, fontweight='bold')

    axes = axes.flatten()
    interaction_results = {}

    plot_idx = 0
    for fatf_var in available_fatf:
        for major_sanction in available_major:
            if plot_idx >= len(axes):
                break

            ax = axes[plot_idx]

            # Create interaction categories
            df['interaction_cat'] = df[fatf_var].astype(str) + '_' + df[major_sanction].astype(str)

            # Create meaningful labels
            interaction_labels = {
                '0_0': 'Neither',
                '0_1': major_sanction.replace('_binary', '').replace('_', ' ').upper() + ' Only',
                '1_0': fatf_var.replace('_binary', '').replace('fatf_', 'FATF ').replace('_', ' ').upper() + ' Only',
                '1_1': 'Both'
            }

            # Get data for each combination
            interaction_data = []
            interaction_means = []
            interaction_ns = []

            for cat in ['0_0', '0_1', '1_0', '1_1']:
                subset = df[df['interaction_cat'] == cat]['score']
                if len(subset) > 0:
                    interaction_data.append(subset.values)
                    interaction_means.append(subset.mean())
                    interaction_ns.append(len(subset))
                else:
                    interaction_data.append([])
                    interaction_means.append(np.nan)
                    interaction_ns.append(0)

            # Create box plot
            valid_data = [data for data in interaction_data if len(data) > 0]
            valid_labels = [interaction_labels[cat] for cat, data in zip(['0_0', '0_1', '1_0', '1_1'], interaction_data)
                            if len(data) > 0]

            if len(valid_data) >= 2:
                box_plot = ax.boxplot(valid_data, labels=valid_labels, patch_artist=True)

                # Color the boxes
                colors = ['lightblue', 'lightcoral', 'lightgreen', 'orange']
                for patch, color in zip(box_plot['boxes'], colors[:len(valid_data)]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)

                # Perform two-way ANOVA
                try:
                    formula = f'score ~ C({fatf_var}) * C({major_sanction})'
                    model = ols(formula, data=df).fit()
                    anova_table = anova_lm(model, typ=2)

                    # Extract interaction p-value
                    interaction_term = f'C({fatf_var}):C({major_sanction})'
                    if interaction_term in anova_table.index:
                        interaction_p = anova_table.loc[interaction_term, 'PR(>F)']
                    else:
                        interaction_p = np.nan

                    interaction_results[f'{fatf_var}_x_{major_sanction}'] = {
                        'anova_table': anova_table,
                        'interaction_p': interaction_p,
                        'means': interaction_means,
                        'sample_sizes': interaction_ns
                    }

                    # Add statistical annotation
                    if not np.isnan(interaction_p):
                        if interaction_p < 0.001:
                            p_text = "Interaction: p < 0.001***"
                        elif interaction_p < 0.01:
                            p_text = f"Interaction: p = {interaction_p:.3f}**"
                        elif interaction_p < 0.05:
                            p_text = f"Interaction: p = {interaction_p:.3f}*"
                        else:
                            p_text = f"Interaction: p = {interaction_p:.3f}"

                        ax.text(0.02, 0.98, p_text, transform=ax.transAxes,
                                fontsize=10, verticalalignment='top',
                                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

                except Exception as e:
                    print(f"ANOVA failed for {fatf_var} x {major_sanction}: {str(e)}")

            else:
                ax.text(0.5, 0.5, 'Insufficient data\nfor interaction analysis',
                        ha='center', va='center', transform=ax.transAxes, fontsize=12)

            # Formatting
            ax.set_ylabel('Risk Score', fontsize=12)
            ax.set_title(
                f'{fatf_var.replace("_binary", "").replace("fatf_", "FATF ").replace("_", " ").upper()}\n× {major_sanction.replace("_binary", "").replace("_", " ").upper()}',
                fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)

            plot_idx += 1

    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig('../../results/figures/03_regulatory/fatf_interaction_effects.png', dpi=300, bbox_inches='tight')
    plt.show()

    return interaction_results


def export_comprehensive_results(df, sanction_vars, individual_results, anova_results,
                                 trend_results, overlap_results, interaction_results):
    """Export all results to Excel"""
    with pd.ExcelWriter('../../results/tables/Analysis3_Regulatory_Impact_Results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: Individual Effects
        individual_summary = []
        for sanction, results in individual_results.items():
            individual_summary.append({
                'Sanction_Type': sanction.replace('_binary', '').replace('_', ' ').upper(),
                'Mean_Not_Sanctioned': results['mean_not_sanctioned'],
                'Mean_Sanctioned': results['mean_sanctioned'],
                'Cohens_d': results['cohens_d'],
                'Cohens_d_CI_Lower': results['cohens_d_ci'][0],
                'Cohens_d_CI_Upper': results['cohens_d_ci'][1],
                'P_Value': results['p_value'],
                'Test_Used': results['test_used'],
                'Effect_Size_Interpretation': results['effect_interpretation'],
                'N_Not_Sanctioned': results['n_not_sanctioned'],
                'N_Sanctioned': results['n_sanctioned']
            })

        individual_df = pd.DataFrame(individual_summary)
        individual_df.to_excel(writer, sheet_name='Individual_Effects', index=False)

        # Sheet 2: FATF Regional
        if anova_results:
            for key, anova_table in anova_results.items():
                anova_table.to_excel(writer, sheet_name='FATF_Regional',
                                     startrow=len(anova_table) * list(anova_results.keys()).index(key) + 2)

        # Sheet 3: Cumulative Impact
        if 'category_means' in trend_results:
            cumulative_df = pd.DataFrame(trend_results['category_means'])
            cumulative_df.to_excel(writer, sheet_name='Cumulative_Impact', index=False)

            # Add correlation results
            correlation_summary = pd.DataFrame({
                'Statistic': ['Linear Correlation', 'P-value', 'R-squared'],
                'Value': [trend_results['linear_correlation'],
                          trend_results['linear_p_value'],
                          trend_results['linear_r_squared']]
            })
            correlation_summary.to_excel(writer, sheet_name='Cumulative_Impact',
                                         startrow=len(cumulative_df) + 3, index=False)

        # Sheet 4: Regulatory Overlap
        if 'sanction_distribution' in overlap_results:
            overlap_df = pd.DataFrame(list(overlap_results['sanction_distribution'].items()),
                                      columns=['Number_of_Sanctions', 'Number_of_Countries'])
            overlap_df.to_excel(writer, sheet_name='Regulatory_Overlap', index=False)

            # Add top combinations
            if overlap_results['top_combinations']:
                combo_data = []
                for combo, count in overlap_results['top_combinations']:
                    combo_str = ' + '.join([s.replace('_binary', '') for s in combo])
                    combo_data.append({'Combination': combo_str, 'Frequency': count})

                combo_df = pd.DataFrame(combo_data)
                combo_df.to_excel(writer, sheet_name='Regulatory_Overlap',
                                  startrow=len(overlap_df) + 3, index=False)

        # Sheet 5: Interaction Analysis
        if interaction_results:
            interaction_summary = []
            for interaction, results in interaction_results.items():
                interaction_summary.append({
                    'Interaction': interaction.replace('_binary', '').replace('_x_', ' × '),
                    'Interaction_P_Value': results.get('interaction_p', np.nan),
                    'Mean_Neither': results['means'][0] if len(results['means']) > 0 else np.nan,
                    'Mean_First_Only': results['means'][1] if len(results['means']) > 1 else np.nan,
                    'Mean_Second_Only': results['means'][2] if len(results['means']) > 2 else np.nan,
                    'Mean_Both': results['means'][3] if len(results['means']) > 3 else np.nan
                })

            interaction_df = pd.DataFrame(interaction_summary)
            interaction_df.to_excel(writer, sheet_name='Interaction_Analysis', index=False)

        # Sheet 6: High Impact Countries
        # Countries with unusual sanction-risk patterns
        if 'sanction_total' in df.columns:
            # Calculate residuals from sanction-risk relationship
            from sklearn.linear_model import LinearRegression
            reg = LinearRegression().fit(df[['sanction_total']], df['score'])
            df['predicted_score'] = reg.predict(df[['sanction_total']])
            df['residual'] = df['score'] - df['predicted_score']

            # Over-performers (lower risk than expected)
            over_performers = df.nsmallest(20, 'residual')[
                ['country', 'sanction_total', 'score', 'predicted_score', 'residual']] if 'country' in df.columns else \
            df.nsmallest(20, 'residual')[['sanction_total', 'score', 'predicted_score', 'residual']]
            over_performers.to_excel(writer, sheet_name='High_Impact_Countries', index=False)

            # Under-performers (higher risk than expected)
            under_performers = df.nlargest(20, 'residual')[
                ['country', 'sanction_total', 'score', 'predicted_score', 'residual']] if 'country' in df.columns else \
            df.nlargest(20, 'residual')[['sanction_total', 'score', 'predicted_score', 'residual']]
            under_performers.to_excel(writer, sheet_name='High_Impact_Countries',
                                      startrow=len(over_performers) + 3, index=False)


def main():
    """Main execution function"""
    print("Starting Basel AML Regulatory Impact & Sanctions Analysis...")

    # Load data
    df, sanction_vars = load_and_prepare_sanctions_data('../../data/processed/basel_aml_with_fsrb.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Analysis will examine {len(sanction_vars)} sanction types: {sanction_vars}")

    # Individual regulatory impact analysis
    print("\n1. Performing individual regulatory impact analysis...")
    individual_results = individual_regulatory_impact_analysis(df, sanction_vars)

    # FATF regional analysis
    print("\n2. Performing FATF regional analysis...")
    anova_results = fatf_regional_analysis(df)

    # Cumulative regulatory impact
    print("\n3. Performing cumulative regulatory impact analysis...")
    trend_results = cumulative_regulatory_impact(df)

    # Regulatory overlap analysis
    print("\n4. Performing regulatory overlap analysis...")
    overlap_results = regulatory_overlap_analysis(df, sanction_vars)

    # FATF interaction effects
    print("\n5. Performing FATF interaction effects analysis...")
    interaction_results = fatf_interaction_effects_analysis(df)

    # Export results
    print("\n6. Exporting results to Excel...")
    export_comprehensive_results(df, sanction_vars, individual_results, anova_results,
                                 trend_results, overlap_results, interaction_results)

    print("\nAnalysis Complete!")
    print("=" * 60)
    print("OUTPUT FILES GENERATED:")
    print("1. individual_regulatory_impact.png - Individual sanction effects (3×4 grid)")
    print("2. effect_size_ranking.png - Cohen's d effect size ranking")
    print("3. fatf_regional_analysis.png - FATF regional impact analysis")
    print("4. cumulative_regulatory_impact.png - Dose-response analysis")
    print("5. regulatory_overlap_analysis.png - Sanction co-occurrence patterns")
    print("6. fatf_interaction_effects.png - FATF interaction effects")
    print("7. Analysis3_Regulatory_Impact_Results.xlsx - Complete statistical tables")
    print("=" * 60)

    # Summary findings
    print(f"\nKEY FINDINGS:")

    # Strongest effect
    if individual_results:
        strongest_effect = max(individual_results.items(), key=lambda x: abs(x[1]['cohens_d']))
        print(f"Strongest Individual Effect: {strongest_effect[0].replace('_binary', '').upper()}")
        print(f"  Cohen's d = {strongest_effect[1]['cohens_d']:.3f} ({strongest_effect[1]['effect_interpretation']})")
        print(f"  p-value = {strongest_effect[1]['p_value']:.6f}")

    # Correlation with sanction total
    if 'linear_correlation' in trend_results:
        print(f"Sanction Total Correlation: r = {trend_results['linear_correlation']:.3f}")
        print(f"  p-value = {trend_results['linear_p_value']:.6f}")

    # Countries with highest sanction burden
    if 'sanction_total' in df.columns:
        max_sanctions = df['sanction_total'].max()
        high_burden_countries = len(df[df['sanction_total'] >= max_sanctions - 1])
        print(f"Countries with Highest Sanction Burden (≥{max_sanctions - 1}): {high_burden_countries}")


if __name__ == "__main__":
    main()