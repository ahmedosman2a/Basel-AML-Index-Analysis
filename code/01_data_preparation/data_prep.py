# Basel AML Index: Complete Data Preparation Implementation
# Focus: Pre-analysis data preparation only

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings

warnings.filterwarnings('ignore')


# =====================================
# 1. DATA LOADING AND INITIAL CLEANING
# =====================================

def load_basel_data(filepath):
    """Load and perform initial cleaning of Basel AML data"""

    # Load data
    df = pd.read_csv(filepath)

    print(f"Initial dataset: {len(df)} records, {df.shape[1]} columns")

    # Remove any completely empty rows
    df = df.dropna(how='all').reset_index(drop=True)

    # Check for missing country names
    missing_countries = df['country'].isnull().sum()
    if missing_countries > 0:
        print(f"Warning: {missing_countries} rows with missing country names")
        df = df.dropna(subset=['country']).reset_index(drop=True)

    # Define variable groups
    numeric_vars = ['score', 'mltf', 'cor', 'fintran', 'pubtran', 'polleg']
    sanction_vars = ['ofac', 'un_sc', 'eu_restrictive_measures', 'eu_non_cooperative_tax',
                     'eu_high_risk', 'uk_sc', 'uk_fs', 'fatf_grey', 'fatf_black',
                     'australian_sc', 'eu_non_cooperative_taxc']

    print(f"Clean dataset: {len(df)} countries")
    return df, numeric_vars, sanction_vars


# =====================================
# 2. MISSING DATA ANALYSIS
# =====================================

def analyze_missing_data(df, numeric_vars):
    """Comprehensive missing data analysis"""

    print("\n=== MISSING DATA ANALYSIS ===")

    # Missing data by variable
    missing_summary = pd.DataFrame({
        'Variable': numeric_vars,
        'Missing_Count': [df[var].isnull().sum() for var in numeric_vars],
        'Missing_Percent': [df[var].isnull().sum() / len(df) * 100 for var in numeric_vars],
        'Valid_Count': [df[var].notna().sum() for var in numeric_vars]
    })

    print("Missing data summary:")
    for _, row in missing_summary.iterrows():
        print(f"{row['Variable']}: {row['Missing_Count']} missing ({row['Missing_Percent']:.1f}%)")

    # Missing data patterns
    missing_matrix = df[numeric_vars].isnull()
    patterns = missing_matrix.apply(lambda x: ''.join(['0' if miss else '1' for miss in x]), axis=1)
    pattern_counts = patterns.value_counts()

    print(f"\nMissing data patterns (1=present, 0=missing):")
    print("Pattern\tCount\tPercent")
    for pattern, count in pattern_counts.head(10).items():
        pct = count / len(df) * 100
        print(f"{pattern}\t{count}\t{pct:.1f}%")

    # Countries with most missing data
    df['missing_count'] = df[numeric_vars].isnull().sum(axis=1)
    most_missing = df[df['missing_count'] > 0].sort_values('missing_count', ascending=False)

    print(f"\nCountries with missing data: {len(most_missing)}/{len(df)}")
    if len(most_missing) > 0:
        print("Most problematic cases:")
        for _, row in most_missing.head(10).iterrows():
            missing_vars = [var for var in numeric_vars if pd.isnull(row[var])]
            print(f"{row['country']}: {row['missing_count']}/6 missing ({', '.join(missing_vars)})")

    return missing_summary, most_missing


# =====================================
# 3. HANDLE ZERO VALUES IN PUBTRAN
# =====================================

def analyze_zero_values(df):
    """Analyze zero values in pubtran variable"""

    print("\n=== ZERO VALUES ANALYSIS ===")

    # Find countries with zero pubtran
    zero_pubtran = df[df['pubtran'] == 0]['country'].tolist()

    print(f"Countries with PUBTRAN = 0: {len(zero_pubtran)}")
    if zero_pubtran:
        print("Countries:", ', '.join(zero_pubtran))

    # Decision helper: Are these likely to be structural zeros?
    print("\nRecommendation: Review these countries manually")
    print("- If they are offshore centers/authoritarian regimes: keep as 0")
    print("- If zeros seem implausible: convert to NaN for imputation")

    return zero_pubtran


def handle_zero_values(df, convert_zeros_to_nan=False):
    """Handle zero values in pubtran based on decision"""

    if convert_zeros_to_nan:
        zero_count = (df['pubtran'] == 0).sum()
        df['pubtran'] = df['pubtran'].replace(0, np.nan)
        print(f"Converted {zero_count} zero values in pubtran to NaN")
    else:
        print("Keeping zero values in pubtran as structural zeros")

    return df


# =====================================
# 4. MULTIPLE IMPUTATION
# =====================================

def multiple_imputation_mice(df, numeric_vars, n_imputations=5):
    """Perform multiple imputation using MICE"""

    print(f"\n=== MULTIPLE IMPUTATION ({n_imputations} datasets) ===")

    # Prepare imputation features
    imputation_data = df[numeric_vars].copy()

    # Add auxiliary variables if available
    # Create total sanction count as auxiliary variable
    sanction_cols = [col for col in df.columns if col in ['ofac', 'un_sc', 'eu_restrictive_measures',
                                                          'fatf_grey', 'fatf_black', 'uk_sc']]
    if sanction_cols:
        # Convert sanctions to binary first
        sanction_binary = df[sanction_cols].applymap(lambda x: 1 if x == 'yes' else 0)
        imputation_data['sanction_total'] = sanction_binary.sum(axis=1)
        print("Added sanction total as auxiliary variable")

    # Perform multiple imputation
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        verbose=1
    )

    imputed_datasets = []
    for i in range(n_imputations):
        print(f"Creating imputed dataset {i + 1}/{n_imputations}")

        # Set different random state for each imputation
        imputer.random_state = 42 + i

        # Perform imputation
        imputed_values = imputer.fit_transform(imputation_data)

        # Create imputed dataframe
        df_imputed = df.copy()
        df_imputed[numeric_vars] = imputed_values[:, :len(numeric_vars)]

        # Validate imputed values are within reasonable bounds (0-10)
        for var in numeric_vars:
            df_imputed[var] = np.clip(df_imputed[var], 0, 10)

        imputed_datasets.append(df_imputed)

    print(f"Multiple imputation completed. Generated {len(imputed_datasets)} datasets.")

    # Use first imputed dataset for further processing
    return imputed_datasets[0], imputed_datasets


# =====================================
# 5. SANCTIONS DATA PREPARATION
# =====================================

def prepare_sanctions_data(df, sanction_vars):
    """Convert sanctions from yes/null to binary and create composite measures"""

    print("\n=== SANCTIONS DATA PREPARATION ===")

    # Convert to binary (yes=1, null=0)
    for var in sanction_vars:
        if var in df.columns:
            original_yes = (df[var] == 'yes').sum()
            df[f'{var}_binary'] = (df[var] == 'yes').astype(int)
            print(f"{var}: {original_yes} countries sanctioned/listed")

    # Create composite measures
    binary_sanction_cols = [f'{var}_binary' for var in sanction_vars if var in df.columns]

    # Total sanction count
    df['sanction_total'] = df[binary_sanction_cols].sum(axis=1)

    # Weighted sanction index
    sanction_weights = {
        'un_sc_binary': 1.0,
        'fatf_black_binary': 0.95,
        'fatf_grey_binary': 0.85,
        'ofac_binary': 0.8,
        'eu_restrictive_measures_binary': 0.8,
        'uk_sc_binary': 0.75,
        'eu_high_risk_binary': 0.7,
        'australian_sc_binary': 0.6,
        'eu_non_cooperative_tax_binary': 0.5,
        'uk_fs_binary': 0.5,
        'eu_non_cooperative_taxc_binary': 0.5
    }

    df['sanction_weighted'] = 0
    for col, weight in sanction_weights.items():
        if col in df.columns:
            df['sanction_weighted'] += df[col] * weight

    # Categorical sanction levels
    df['sanction_level'] = pd.cut(
        df['sanction_total'],
        bins=[-0.1, 0, 2, 4, float('inf')],
        labels=['None', 'Low', 'Medium', 'High']
    )

    # Summary
    print(f"\nSanction summary:")
    print(f"- Total sanctions range: {df['sanction_total'].min()}-{df['sanction_total'].max()}")
    print(f"- Weighted sanctions range: {df['sanction_weighted'].min():.2f}-{df['sanction_weighted'].max():.2f}")
    print(f"- Sanction level distribution:")
    print(df['sanction_level'].value_counts())

    return df


# =====================================
# 6. NORMALIZATION
# =====================================

def normalize_numeric_variables(df, numeric_vars, method='robust'):
    """Normalize numeric variables using specified method"""

    print(f"\n=== NORMALIZATION ({method.upper()}) ===")

    if method == 'robust':
        scaler = RobustScaler()
        suffix = '_robust'
    elif method == 'standard':
        scaler = StandardScaler()
        suffix = '_std'
    else:
        raise ValueError("Method must be 'robust' or 'standard'")

    # Apply normalization
    normalized_values = scaler.fit_transform(df[numeric_vars])

    # Create normalized columns
    for i, var in enumerate(numeric_vars):
        df[f'{var}{suffix}'] = normalized_values[:, i]

    # Summary statistics
    print("Normalization summary:")
    for var in numeric_vars:
        norm_var = f'{var}{suffix}'
        print(f"{var}: mean={df[norm_var].mean():.3f}, std={df[norm_var].std():.3f}")

    return df, scaler


# =====================================
# 7. OUTLIER DETECTION
# =====================================

def detect_outliers(df, numeric_vars, method='iqr'):
    """Detect outliers using IQR method"""

    print(f"\n=== OUTLIER DETECTION ({method.upper()}) ===")

    outlier_countries = set()
    outlier_details = []

    for var in numeric_vars:
        if method == 'iqr':
            Q1 = df[var].quantile(0.25)
            Q3 = df[var].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)]

        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(df[var].dropna()))
            outliers = df[z_scores > 3]

        if len(outliers) > 0:
            print(f"{var}: {len(outliers)} outliers")
            for _, row in outliers.iterrows():
                outlier_countries.add(row['country'])
                outlier_details.append({
                    'country': row['country'],
                    'variable': var,
                    'value': row[var],
                    'bounds': f"[{lower_bound:.2f}, {upper_bound:.2f}]" if method == 'iqr' else 'z>3'
                })

    print(f"\nCountries with outliers: {len(outlier_countries)}")
    if outlier_countries:
        print("Countries:", ', '.join(list(outlier_countries)[:10]))

    return outlier_details, outlier_countries


# =====================================
# 8. QUALITY VALIDATION
# =====================================

def validate_data_quality(df, numeric_vars):
    """Comprehensive data quality validation"""

    print("\n=== DATA QUALITY VALIDATION ===")

    # Check for remaining missing values
    missing_check = df[numeric_vars].isnull().sum()
    if missing_check.sum() > 0:
        print("WARNING: Still have missing values after imputation")
        print(missing_check[missing_check > 0])
    else:
        print("✓ No missing values in numeric variables")

    # Check value ranges (should be 0-10 for Basel scores)
    for var in numeric_vars:
        min_val = df[var].min()
        max_val = df[var].max()
        if min_val < 0 or max_val > 10:
            print(f"WARNING: {var} values outside 0-10 range: [{min_val:.2f}, {max_val:.2f}]")
        else:
            print(f"✓ {var} values within valid range: [{min_val:.2f}, {max_val:.2f}]")

    # Check for duplicates
    duplicates = df['country'].duplicated().sum()
    if duplicates > 0:
        print(f"WARNING: {duplicates} duplicate countries")
    else:
        print("✓ No duplicate countries")

    # Distribution check
    print("\nDistribution summary:")
    print(df[numeric_vars].describe().round(2))

    return True


# =====================================
# 9. MAIN PREPARATION PIPELINE
# =====================================

def run_data_preparation_pipeline(filepath, convert_zeros_to_nan=False, normalization_method='robust'):
    """
    Complete data preparation pipeline for Basel AML index

    Parameters:
    filepath: path to CSV file
    convert_zeros_to_nan: whether to treat pubtran zeros as missing
    normalization_method: 'robust' or 'standard'
    """

    print("=" * 60)
    print("BASEL AML INDEX DATA PREPARATION PIPELINE")
    print("=" * 60)

    # Step 1: Load data
    df, numeric_vars, sanction_vars = load_basel_data(filepath)

    # Step 2: Analyze missing data
    missing_summary, problematic_countries = analyze_missing_data(df, numeric_vars)

    # Step 3: Handle zero values
    zero_countries = analyze_zero_values(df)
    df = handle_zero_values(df, convert_zeros_to_nan)

    # Step 4: Multiple imputation (if needed)
    if df[numeric_vars].isnull().sum().sum() > 0:
        df_imputed, all_imputed = multiple_imputation_mice(df, numeric_vars)
        df = df_imputed  # Use first imputed dataset for pipeline
    else:
        print("No missing data - skipping imputation")
        all_imputed = [df]

    # Step 5: Prepare sanctions data
    df = prepare_sanctions_data(df, sanction_vars)

    # Step 6: Normalize numeric variables
    df, scaler = normalize_numeric_variables(df, numeric_vars, normalization_method)

    # Step 7: Detect outliers
    outlier_details, outlier_countries = detect_outliers(df, numeric_vars)

    # Step 8: Validate data quality
    validate_data_quality(df, numeric_vars)

    # Final summary
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETED")
    print("=" * 60)
    print(f"Final dataset: {len(df)} countries")
    print(f"Variables prepared: {len(numeric_vars)} numeric + sanctions")
    print(f"Imputed datasets available: {len(all_imputed)}")
    print(f"Countries with outliers: {len(outlier_countries)}")

    # Return comprehensive results
    results = {
        'data': df,
        'numeric_vars': numeric_vars,
        'sanction_vars': sanction_vars,
        'missing_summary': missing_summary,
        'zero_countries': zero_countries,
        'imputed_datasets': all_imputed,
        'outlier_details': outlier_details,
        'outlier_countries': outlier_countries,
        'scaler': scaler
    }

    return results


# =====================================
# 10. EXPORT ANALYSIS-READY DATA
# =====================================

def export_analysis_ready_data(results, output_path):
    """Export clean, analysis-ready dataset"""

    df = results['data']
    numeric_vars = results['numeric_vars']

    # Select key columns for analysis
    analysis_columns = ['country', 'ISO2'] + numeric_vars

    # Add normalized versions
    norm_suffix = '_robust' if 'score_robust' in df.columns else '_std'
    normalized_vars = [f'{var}{norm_suffix}' for var in numeric_vars]
    analysis_columns.extend(normalized_vars)

    # Add sanction measures
    sanction_measures = ['sanction_total', 'sanction_weighted', 'sanction_level']
    analysis_columns.extend(sanction_measures)

    # Add binary sanction variables
    binary_sanctions = [col for col in df.columns if col.endswith('_binary')]
    analysis_columns.extend(binary_sanctions)

    # Create final dataset
    df_final = df[analysis_columns].copy()

    # Export
    df_final.to_csv(output_path, index=False)
    print(f"\nAnalysis-ready dataset exported to: {output_path}")
    print(f"Dataset shape: {df_final.shape}")
    print(f"Columns: {list(df_final.columns)}")

    return df_final


# =====================================
# 11. USAGE EXAMPLE
# =====================================

if __name__ == "__main__":
    # Example usage
    filepath = "../../data/basel-aml-index-expertedition.csv"

    # Run complete preparation pipeline
    results = run_data_preparation_pipeline(
        filepath=filepath,
        convert_zeros_to_nan=False,  # Keep zeros as structural zeros
        normalization_method='robust'  # Use robust scaling
    )

    # Export analysis-ready data
    final_data = export_analysis_ready_data(results, "basel_aml_analysis_ready.csv")

    print("\n" + "=" * 60)
    print("DATA PREPARATION PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print("Next steps:")
    print("1. Review outlier countries and decide on treatment")
    print("2. Conduct sensitivity analysis with different imputation datasets")
    print("3. Begin statistical analysis with prepared data")
    print("4. Document all preparation decisions for reproducibility")