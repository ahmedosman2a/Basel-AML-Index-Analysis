# Required imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, shapiro, anderson, kstest
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import warnings

warnings.filterwarnings('ignore')

# Set style and color palettes
plt.style.use('default')
sns.set_palette("Set2")


def load_and_preprocess_data(file_path):
    """Load and preprocess the Basel AML dataset"""
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully: {df.shape[0]} countries, {df.shape[1]} variables")

        # Check for required columns
        required_cols = ['score', 'mltf', 'cor', 'fintran', 'pubtran', 'polleg', 'primary_fsrb']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Warning: Missing columns: {missing_cols}")

        # Remove rows with missing score values
        initial_count = len(df)
        df = df.dropna(subset=['score'])
        final_count = len(df)
        if initial_count != final_count:
            print(f"Removed {initial_count - final_count} rows with missing scores")

        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None


def create_risk_categories(df):
    """Create risk categories based on score percentiles"""
    percentiles = df['score'].quantile([0.25, 0.50, 0.75]).values

    def categorize_risk(score):
        if score <= percentiles[0]:
            return 'Low Risk'
        elif score <= percentiles[1]:
            return 'Medium Risk'
        elif score <= percentiles[2]:
            return 'High Risk'
        else:
            return 'Very High Risk'

    df['risk_category'] = df['score'].apply(categorize_risk)

    # Define colors for categories
    risk_colors = {
        'Low Risk': '#2E8B57',      # Green=Low
        'Medium Risk': '#FFD700',   # Yellow=Medium
        'High Risk': '#FF8C00',     # Orange=High
        'Very High Risk': '#DC143C' # Red=Very High
    }

    return df, risk_colors, percentiles


def global_distribution_analysis(df):
    """Create comprehensive distribution dashboard"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Global AML Risk Score Distribution Dashboard', fontsize=16, fontweight='bold')

    # Calculate statistics
    mean_score = df['score'].mean()
    median_score = df['score'].median()
    mode_score = df['score'].mode().iloc[0] if not df['score'].mode().empty else median_score
    std_score = df['score'].std()

    # Top-left: Histogram with density curve
    ax1 = axes[0, 0]
    ax1.hist(df['score'], bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.axvline(mean_score, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_score:.2f}')
    ax1.axvline(median_score, color='green', linestyle='--', linewidth=2, label=f'Median: {median_score:.2f}')
    ax1.axvline(mode_score, color='purple', linestyle='--', linewidth=2, label=f'Mode: {mode_score:.2f}')

    # Add density curve
    x = np.linspace(df['score'].min(), df['score'].max(), 100)
    kde = stats.gaussian_kde(df['score'])
    ax1.plot(x, kde(x), 'navy', linewidth=2, label='Density')

    ax1.set_title('Distribution with Density Curve', fontsize=14)
    ax1.set_xlabel('Risk Score', fontsize=12)
    ax1.set_ylabel('Density', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Top-right: Box plot
    ax2 = axes[0, 1]
    box_plot = ax2.boxplot(df['score'], patch_artist=True, labels=['Risk Score'])
    box_plot['boxes'][0].set_facecolor('lightblue')

    # Add statistical annotations
    q1, q3 = df['score'].quantile([0.25, 0.75])
    iqr = q3 - q1
    ax2.text(1.1, mean_score, f'Mean: {mean_score:.2f}', fontsize=10, ha='left')
    ax2.text(1.1, median_score, f'Median: {median_score:.2f}', fontsize=10, ha='left')
    ax2.text(1.1, q1, f'Q1: {q1:.2f}', fontsize=10, ha='left')
    ax2.text(1.1, q3, f'Q3: {q3:.2f}', fontsize=10, ha='left')
    ax2.text(1.1, std_score + mean_score, f'Std: {std_score:.2f}', fontsize=10, ha='left')

    ax2.set_title('Box Plot with Quartiles', fontsize=14)
    ax2.set_ylabel('Risk Score', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Bottom-left: Violin plot
    ax3 = axes[1, 0]
    violin_parts = ax3.violinplot([df['score']], positions=[1], showmedians=True, showextrema=True)
    for pc in violin_parts['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)

    ax3.set_title('Violin Plot with Quartiles', fontsize=14)
    ax3.set_ylabel('Risk Score', fontsize=12)
    ax3.set_xticks([1])
    ax3.set_xticklabels(['Risk Score'])
    ax3.grid(True, alpha=0.3)

    # Bottom-right: Q-Q plot
    ax4 = axes[1, 1]
    stats.probplot(df['score'], dist="norm", plot=ax4)
    ax4.set_title('Q-Q Plot for Normality Assessment', fontsize=14)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/01_descriptive/risk_distribution_dashboard.png', dpi=300, bbox_inches='tight')
    plt.show()

    return mean_score, median_score, std_score


def create_risk_category_visualizations(df, risk_colors):
    """Create risk category pie and bar charts"""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Risk Category Distribution Analysis', fontsize=16, fontweight='bold')

    # Get category counts
    category_counts = df['risk_category'].value_counts()
    category_percentages = df['risk_category'].value_counts(normalize=True) * 100

    # Pie chart
    ax1 = axes[0]
    colors = [risk_colors[cat] for cat in category_counts.index]
    wedges, texts, autotexts = ax1.pie(category_counts.values, labels=category_counts.index,
                                       autopct=lambda pct: f'{pct:.1f}%\n({int(pct / 100 * len(df))} countries)',
                                       colors=colors, startangle=90)

    ax1.set_title('Risk Category Distribution', fontsize=14)

    # Horizontal bar chart
    ax2 = axes[1]
    bars = ax2.barh(range(len(category_counts)), category_counts.values,
                    color=[risk_colors[cat] for cat in category_counts.index])
    ax2.set_yticks(range(len(category_counts)))
    ax2.set_yticklabels(category_counts.index)
    ax2.set_xlabel('Number of Countries', fontsize=12)
    ax2.set_title('Risk Category Counts', fontsize=14)

    # Add value labels on bars
    for i, (bar, count) in enumerate(zip(bars, category_counts.values)):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f'{count}', ha='left', va='center', fontsize=10)

    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('../../results/figures/01_descriptive/risk_categories_charts.png', dpi=300, bbox_inches='tight')
    plt.show()

    return category_counts, category_percentages


def regional_risk_analysis(df):
    """Perform comprehensive regional analysis"""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Regional Risk Analysis', fontsize=16, fontweight='bold')

    # Remove regions with too few countries (less than 3)
    region_counts = df['primary_fsrb'].value_counts()
    valid_regions = region_counts[region_counts >= 3].index
    df_regional = df[df['primary_fsrb'].isin(valid_regions)].copy()

    # Box plots by region
    ax1 = axes[0, 0]
    sns.boxplot(data=df_regional, x='primary_fsrb', y='score', ax=ax1)
    ax1.set_title('Risk Score Distribution by Region', fontsize=14)
    ax1.set_xlabel('Region (FSRB)', fontsize=12)
    ax1.set_ylabel('Risk Score', fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)

    # Bar chart with confidence intervals
    ax2 = axes[0, 1]
    regional_means = df_regional.groupby('primary_fsrb')['score'].agg(['mean', 'std', 'count']).reset_index()
    regional_means['ci'] = 1.96 * regional_means['std'] / np.sqrt(regional_means['count'])

    bars = ax2.bar(range(len(regional_means)), regional_means['mean'],
                   yerr=regional_means['ci'], capsize=5, alpha=0.7)
    ax2.set_xticks(range(len(regional_means)))
    ax2.set_xticklabels(regional_means['primary_fsrb'], rotation=45)
    ax2.set_title('Mean Risk Score by Region (95% CI)', fontsize=14)
    ax2.set_xlabel('Region (FSRB)', fontsize=12)
    ax2.set_ylabel('Mean Risk Score', fontsize=12)
    ax2.grid(True, alpha=0.3)

    # Violin plots by region
    ax3 = axes[1, 0]
    sns.violinplot(data=df_regional, x='primary_fsrb', y='score', ax=ax3)
    ax3.set_title('Risk Score Distributions by Region', fontsize=14)
    ax3.set_xlabel('Region (FSRB)', fontsize=12)
    ax3.set_ylabel('Risk Score', fontsize=12)
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)

    # Regional risk category distribution
    ax4 = axes[1, 1]
    region_risk_crosstab = pd.crosstab(
        df_regional['primary_fsrb'], df_regional['risk_category'],
        normalize='index'
    ) * 100

    # Ensure correct category order
    category_order = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
    region_risk_crosstab = region_risk_crosstab.reindex(columns=category_order, fill_value=0)

    # Plot with explicit color mapping
    # Get risk_colors from the global namespace or by recomputing
    risk_colors = {
        'Low Risk': '#2E8B57',
        'Medium Risk': '#FFD700',
        'High Risk': '#FF8C00',
        'Very High Risk': '#DC143C'
    }
    region_risk_crosstab.plot(
        kind='bar', stacked=True, ax=ax4,
        color=[risk_colors[cat] for cat in category_order]
    )
    ax4.set_title('Risk Category Distribution by Region (%)', fontsize=14)
    ax4.set_xlabel('Region (FSRB)', fontsize=12)
    ax4.set_ylabel('Percentage', fontsize=12)
    ax4.tick_params(axis='x', rotation=45)
    ax4.legend(title='Risk Category', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../../results/figures/01_descriptive/regional_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Statistical tests
    regional_groups = [group['score'].values for name, group in df_regional.groupby('primary_fsrb')]

    # One-way ANOVA
    f_stat, p_value = f_oneway(*regional_groups)

    # Effect size (eta-squared)
    model = ols('score ~ C(primary_fsrb)', data=df_regional).fit()
    anova_table = anova_lm(model, typ=2)
    eta_squared = anova_table['sum_sq'][0] / anova_table['sum_sq'].sum()

    # Post-hoc Tukey HSD
    tukey_results = pairwise_tukeyhsd(df_regional['score'], df_regional['primary_fsrb'], alpha=0.05)

    return f_stat, p_value, eta_squared, tukey_results, regional_means


def create_top_bottom_countries_viz(df):
    """Create visualizations for highest and lowest risk countries"""
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    fig.suptitle('Top and Bottom Risk Countries', fontsize=16, fontweight='bold')

    # Top 20 highest risk countries
    top_20_high = df.nlargest(20, 'score')
    ax1 = axes[0]
    bars_high = ax1.barh(range(len(top_20_high)), top_20_high['score'],
                         color=plt.cm.Reds(np.linspace(0.4, 1, len(top_20_high))))
    ax1.set_yticks(range(len(top_20_high)))
    ax1.set_yticklabels(top_20_high['country'] if 'country' in df.columns else top_20_high.index)
    ax1.set_xlabel('Risk Score', fontsize=12)
    ax1.set_title('Top 20 Highest Risk Countries', fontsize=14)
    ax1.invert_yaxis()

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars_high, top_20_high['score'])):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{score:.2f}', ha='left', va='center', fontsize=9)

    ax1.grid(True, alpha=0.3, axis='x')

    # Top 20 lowest risk countries
    top_20_low = df.nsmallest(20, 'score')
    ax2 = axes[1]
    bars_low = ax2.barh(range(len(top_20_low)), top_20_low['score'],
                        color=plt.cm.Greens(np.linspace(0.4, 1, len(top_20_low))))
    ax2.set_yticks(range(len(top_20_low)))
    ax2.set_yticklabels(top_20_low['country'] if 'country' in df.columns else top_20_low.index)
    ax2.set_xlabel('Risk Score', fontsize=12)
    ax2.set_title('Top 20 Lowest Risk Countries', fontsize=14)
    ax2.invert_yaxis()

    # Add score labels
    for i, (bar, score) in enumerate(zip(bars_low, top_20_low['score'])):
        ax2.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                 f'{score:.2f}', ha='left', va='center', fontsize=9)

    ax2.grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig('../../results/figures/01_descriptive/top_bottom_countries.png', dpi=300, bbox_inches='tight')
    plt.show()

    return top_20_high, top_20_low


def perform_normality_tests(data):
    """Perform comprehensive normality testing"""
    # Shapiro-Wilk test (for n < 5000)
    if len(data) < 5000:
        shapiro_stat, shapiro_p = shapiro(data)
    else:
        shapiro_stat, shapiro_p = np.nan, np.nan

    # Anderson-Darling test
    anderson_result = anderson(data, dist='norm')
    anderson_stat = anderson_result.statistic
    anderson_critical = anderson_result.critical_values[2]  # 5% significance level
    anderson_p = "< 0.05" if anderson_stat > anderson_critical else "> 0.05"

    # Kolmogorov-Smirnov test
    ks_stat, ks_p = kstest(data, 'norm', args=(data.mean(), data.std()))

    return {
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'anderson_stat': anderson_stat,
        'anderson_p': anderson_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p
    }


def export_to_excel(df, risk_colors, percentiles, normality_results, regional_stats, tukey_results):
    """Export comprehensive results to Excel"""
    with pd.ExcelWriter('../../results/tables/Analysis1_Risk_Distribution_Results.xlsx', engine='openpyxl') as writer:

        # Sheet 1: Global Statistics
        global_stats = pd.DataFrame({
            'Statistic': ['Count', 'Mean', 'Median', 'Mode', 'Standard Deviation', 'Variance',
                          'Skewness', 'Kurtosis', 'Minimum', 'Maximum', 'Range',
                          '5th Percentile', '10th Percentile', '25th Percentile',
                          '50th Percentile', '75th Percentile', '90th Percentile', '95th Percentile'],
            'Value': [
                len(df),
                df['score'].mean(),
                df['score'].median(),
                df['score'].mode().iloc[0] if not df['score'].mode().empty else df['score'].median(),
                df['score'].std(),
                df['score'].var(),
                df['score'].skew(),
                df['score'].kurtosis(),
                df['score'].min(),
                df['score'].max(),
                df['score'].max() - df['score'].min(),
                df['score'].quantile(0.05),
                df['score'].quantile(0.10),
                df['score'].quantile(0.25),
                df['score'].quantile(0.50),
                df['score'].quantile(0.75),
                df['score'].quantile(0.90),
                df['score'].quantile(0.95)
            ]
        })

        # Add normality test results
        normality_df = pd.DataFrame({
            'Test': ['Shapiro-Wilk', 'Anderson-Darling', 'Kolmogorov-Smirnov'],
            'Statistic': [normality_results['shapiro_stat'],
                          normality_results['anderson_stat'],
                          normality_results['ks_stat']],
            'P-value': [normality_results['shapiro_p'],
                        normality_results['anderson_p'],
                        normality_results['ks_p']]
        })

        global_stats.to_excel(writer, sheet_name='Global_Statistics', index=False, startrow=0)
        normality_df.to_excel(writer, sheet_name='Global_Statistics', index=False, startrow=len(global_stats) + 3)

        # Sheets 2-5: Risk Categories
        risk_categories = ['Low Risk', 'Medium Risk', 'High Risk', 'Very High Risk']
        for category in risk_categories:
            category_data = df[df['risk_category'] == category]
            if not category_data.empty:
                sheet_name = category.replace(' ', '_') + '_Countries'
                export_cols = ['country', 'score', 'primary_fsrb'] if 'country' in df.columns else ['score',
                                                                                                    'primary_fsrb']
                available_cols = [col for col in export_cols if col in df.columns]
                category_data[available_cols].to_excel(writer, sheet_name=sheet_name, index=False)

        # Sheet 6: Risk Categories Summary
        category_summary = df['risk_category'].value_counts().reset_index()
        category_summary.columns = ['Risk_Category', 'Count']
        category_summary['Percentage'] = (category_summary['Count'] / len(df)) * 100
        category_summary['Mean_Score'] = [df[df['risk_category'] == cat]['score'].mean()
                                          for cat in category_summary['Risk_Category']]
        category_summary['Category_Boundaries'] = [
            f"0 - {percentiles[0]:.2f}",
            f"{percentiles[0]:.2f} - {percentiles[1]:.2f}",
            f"{percentiles[1]:.2f} - {percentiles[2]:.2f}",
            f"{percentiles[2]:.2f} - {df['score'].max():.2f}"
        ]
        category_summary.to_excel(writer, sheet_name='Risk_Categories_Summary', index=False)

        # Sheet 7: Regional Analysis
        if regional_stats is not None:
            regional_stats.to_excel(writer, sheet_name='Regional_Statistics', index=False)

        # Sheet 8: Tukey HSD Results
        if tukey_results is not None:
            # Use the public summary data instead of private attributes
            tukey_df = pd.DataFrame(data=tukey_results.summary().data[1:],
                                    columns=tukey_results.summary().data[0])
            tukey_df.to_excel(writer, sheet_name='Tukey_HSD_Results', index=False)


def main():
    """Main execution function"""
    print("Starting Basel AML Risk Distribution Analysis...")

    # Load data
    df = load_and_preprocess_data('../../data/processed/basel_aml_with_fsrb.csv')
    if df is None:
        print("Failed to load data. Exiting.")
        return

    print(f"Analysis will cover {len(df)} countries")

    # Create risk categories
    df, risk_colors, percentiles = create_risk_categories(df)
    print("Risk categories created based on quartiles")

    # Global distribution analysis
    print("Performing global distribution analysis...")
    mean_score, median_score, std_score = global_distribution_analysis(df)

    # Risk category visualizations
    print("Creating risk category visualizations...")
    category_counts, category_percentages = create_risk_category_visualizations(df, risk_colors)

    # Normality tests
    print("Performing normality tests...")
    normality_results = perform_normality_tests(df['score'])

    # Regional analysis
    print("Performing regional analysis...")
    if 'primary_fsrb' in df.columns:
        f_stat, p_value, eta_squared, tukey_results, regional_stats = regional_risk_analysis(df)
        print(f"Regional ANOVA: F={f_stat:.3f}, p={p_value:.6f}, η²={eta_squared:.3f}")
    else:
        print("Warning: 'primary_fsrb' column not found. Skipping regional analysis.")
        regional_stats, tukey_results = None, None

    # Top/Bottom countries
    print("Creating top/bottom countries visualization...")
    top_20_high, top_20_low = create_top_bottom_countries_viz(df)

    # Export to Excel
    print("Exporting results to Excel...")
    export_to_excel(df, risk_colors, percentiles, normality_results, regional_stats, tukey_results)

    print("\nAnalysis Complete!")
    print("=" * 50)
    print("OUTPUT FILES GENERATED:")
    print("1. risk_distribution_dashboard.png - Distribution analysis dashboard")
    print("2. risk_categories_charts.png - Risk category visualizations")
    print("3. regional_analysis.png - Regional comparison analysis")
    print("4. top_bottom_countries.png - Highest/lowest risk countries")
    print("5. Analysis1_Risk_Distribution_Results.xlsx - Complete statistical results")
    print("=" * 50)

    # Summary statistics
    print(f"\nSUMMARY STATISTICS:")
    print(f"Total Countries: {len(df)}")
    print(f"Mean Risk Score: {mean_score:.3f}")
    print(f"Median Risk Score: {median_score:.3f}")
    print(f"Standard Deviation: {std_score:.3f}")
    print(f"\nRisk Category Distribution:")
    for category, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"  {category}: {count} countries ({percentage:.1f}%)")


if __name__ == "__main__":
    main()