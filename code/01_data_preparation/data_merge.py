#!/usr/bin/env python3
"""
Basel AML Index + FSRB Regions Data Merger
==========================================

This script combines your processed Basel AML data with FSRB regional classifications.

USAGE:
1. Save this script as 'merge_basel_fsrb.py'
2. Update the file paths below (lines 25-27)
3. Run: python merge_basel_fsrb.py

OUTPUT:
- Combined CSV with all Basel AML columns + FSRB classifications
- Handles multiple FSRB memberships
- Creates binary columns for each FSRB body
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS TO YOUR ACTUAL FILES
# =============================================================================

BASEL_FILE = "../../data/basel_aml_analysis_ready.csv"  # Your processed Basel AML data
FSRB_FILE = "../../data/fsrb_regions.csv"  # Your FSRB regions data
OUTPUT_FILE = "../../data/processed/basel_aml_with_fsrb.csv"  # Output combined file


# =============================================================================
# MAIN MERGER FUNCTION
# =============================================================================

def merge_basel_fsrb():
    """Merge Basel AML data with FSRB regional classifications"""

    print("🔄 Loading datasets...")

    # Load data
    try:
        basel_df = pd.read_csv(BASEL_FILE)
        fsrb_df = pd.read_csv(FSRB_FILE)
    except FileNotFoundError as e:
        print(f"❌ Error: Could not find file - {e}")
        print("💡 Please update the file paths at the top of this script")
        return None

    print(f"✅ Basel AML data: {len(basel_df)} countries, {len(basel_df.columns)} columns")
    print(f"✅ FSRB data: {len(fsrb_df)} countries, {len(fsrb_df.columns)} columns")

    # Clean column names
    fsrb_df.columns = fsrb_df.columns.str.strip()

    # Country name standardization
    def standardize_name(name):
        if pd.isna(name):
            return name
        return str(name).strip().replace('  ', ' ')

    basel_df['country_std'] = basel_df['country'].apply(standardize_name)
    fsrb_df['country_std'] = fsrb_df['Country'].apply(standardize_name)

    # Manual country name mappings for common mismatches
    name_corrections = {
        'United States': 'United States of America',
        'Russia': 'Russian Federation',
        'South Korea': 'Korea, Republic of',
        'North Korea': "Korea, Democratic People's Republic of",
        'Iran': 'Iran, Islamic Republic of',
        'Syria': 'Syrian Arab Republic',
        'Venezuela': 'Venezuela, Bolivarian Republic of',
        'Tanzania': 'Tanzania, United Republic of',
        'Moldova': 'Moldova, Republic of',
        'Vietnam': 'Viet Nam',
        'Laos': "Lao People's Democratic Republic",
        'Cape Verde': 'Cabo Verde',
        'Brunei Darussalam': 'Brunei',
        'Bolivia': 'Bolivia, Plurinational State of'
    }

    basel_df['country_corrected'] = basel_df['country_std'].replace(name_corrections)

    print("🔄 Matching countries...")

    # Step 1: ISO code matching
    merged_df = pd.merge(basel_df, fsrb_df,
                         left_on='ISO2', right_on='ISO Code',
                         how='left', suffixes=('', '_fsrb'))

    iso_matches = merged_df['Country'].notna().sum()
    print(f"📍 ISO code matches: {iso_matches}")

    # Step 2: Country name matching for unmatched countries
    unmatched_mask = merged_df['Country'].isna()

    for idx in merged_df[unmatched_mask].index:
        basel_country = merged_df.at[idx, 'country_corrected']

        # Try exact name match
        fsrb_match = fsrb_df[fsrb_df['country_std'] == basel_country]

        if len(fsrb_match) == 1:
            # Found exact match
            for col in fsrb_df.columns:
                if col != 'country_std':
                    merged_df.at[idx, col] = fsrb_match.iloc[0][col]
        else:
            # Try partial match
            partial_match = fsrb_df[fsrb_df['country_std'].str.contains(basel_country, case=False, na=False)]
            if len(partial_match) == 1:
                for col in fsrb_df.columns:
                    if col != 'country_std':
                        merged_df.at[idx, col] = partial_match.iloc[0][col]

    total_matches = merged_df['Country'].notna().sum()
    print(f"🎯 Total matches: {total_matches}/{len(basel_df)} ({total_matches / len(basel_df) * 100:.1f}%)")

    # Clean up temporary columns
    merged_df = merged_df.drop(columns=['country_std', 'country_corrected'], errors='ignore')

    # Rename FSRB columns for clarity
    column_renames = {
        'Country': 'fsrb_country_name',
        'ISO Code': 'fsrb_iso_code',
        'FATF-Style Regional Bodies (FSRB)': 'fsrb_memberships'
    }
    merged_df = merged_df.rename(columns=column_renames)

    print("🔄 Processing FSRB memberships...")

    # Process multiple FSRB memberships
    if 'fsrb_memberships' in merged_df.columns:

        # Count number of FSRB memberships per country
        merged_df['fsrb_count'] = merged_df['fsrb_memberships'].apply(
            lambda x: len(str(x).split(';')) if pd.notna(x) else 0
        )

        # Extract primary (first listed) FSRB
        merged_df['primary_fsrb'] = merged_df['fsrb_memberships'].apply(
            lambda x: str(x).split(';')[0].strip() if pd.notna(x) else None
        )

        # Get all unique FSRB bodies for binary columns
        all_fsrbs = set()
        for membership in merged_df['fsrb_memberships'].dropna():
            if ';' in str(membership):
                fsrbs = [fsrb.strip() for fsrb in str(membership).split(';')]
            else:
                fsrbs = [str(membership).strip()]
            all_fsrbs.update(fsrbs)

        all_fsrbs = sorted(list(all_fsrbs))
        print(f"📊 Found {len(all_fsrbs)} unique FSRB bodies")

        # Create binary columns for each FSRB body
        for fsrb in all_fsrbs:
            # Create clean column name
            col_name = f"fsrb_{fsrb.lower().replace(' ', '_').replace('-', '_').replace('(', '').replace(')', '').replace('/', '_').replace(',', '').replace("'", '')}"

            # Create binary indicator (1 if country is member, 0 if not)
            merged_df[col_name] = merged_df['fsrb_memberships'].apply(
                lambda x: 1 if pd.notna(x) and fsrb in str(x) else 0
            )

            member_count = merged_df[col_name].sum()
            print(f"   • {fsrb}: {member_count} countries")

        # Show FSRB distribution
        print("\n📈 Primary FSRB distribution:")
        fsrb_dist = merged_df['primary_fsrb'].value_counts()
        for fsrb, count in fsrb_dist.items():
            print(f"   • {fsrb}: {count} countries")

        # Show countries with multiple memberships
        multiple_members = merged_df[merged_df['fsrb_count'] > 1]
        if len(multiple_members) > 0:
            print(f"\n🔗 Countries with multiple FSRB memberships ({len(multiple_members)}):")
            for _, row in multiple_members[['country', 'fsrb_memberships']].iterrows():
                print(f"   • {row['country']}: {row['fsrb_memberships']}")

    # Add indicator for whether country has FSRB data
    merged_df['has_fsrb_data'] = merged_df['fsrb_country_name'].notna()

    # Show unmatched countries
    unmatched = merged_df[~merged_df['has_fsrb_data']]
    if len(unmatched) > 0:
        print(f"\n⚠️  Unmatched countries ({len(unmatched)}):")
        for country in unmatched['country'].tolist():
            print(f"   • {country}")

    print("💾 Saving combined dataset...")

    # Save the final dataset
    merged_df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n✅ SUCCESS! Combined dataset created:")
    print(f"   📁 File: {OUTPUT_FILE}")
    print(f"   📊 Countries: {len(merged_df)}")
    print(f"   📈 Columns: {len(merged_df.columns)}")
    print(f"   🎯 With FSRB data: {merged_df['has_fsrb_data'].sum()}")

    # Show final column structure
    print(f"\n📋 Final dataset columns:")
    original_cols = [col for col in merged_df.columns if
                     not col.startswith('fsrb_') and col not in ['has_fsrb_data', 'fsrb_country_name', 'fsrb_iso_code',
                                                                 'fsrb_memberships', 'fsrb_count', 'primary_fsrb']]
    fsrb_cols = [col for col in merged_df.columns if
                 col.startswith('fsrb_') or col in ['fsrb_country_name', 'fsrb_iso_code', 'fsrb_memberships',
                                                    'fsrb_count', 'primary_fsrb', 'has_fsrb_data']]

    print(f"   🏛️  Original Basel AML columns: {len(original_cols)}")
    print(f"   🌍 FSRB-related columns: {len(fsrb_cols)}")

    return merged_df


# =============================================================================
# EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("   BASEL AML INDEX + FSRB REGIONS MERGER")
    print("=" * 60)

    result = merge_basel_fsrb()

    if result is not None:
        print("\n" + "=" * 60)
        print("🎉 MERGER COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"\nYour combined dataset is ready: {OUTPUT_FILE}")
        print(f"Shape: {result.shape[0]} countries × {result.shape[1]} columns")

        # Quick data preview
        print(f"\n📋 Sample of combined data:")
        preview_cols = ['country', 'score', 'sanction_total', 'primary_fsrb', 'fsrb_count']
        available_cols = [col for col in preview_cols if col in result.columns]
        print(result[available_cols].head().to_string(index=False))

    else:
        print("\n❌ Merger failed. Please check file paths and try again.")
        print(f"Expected files:")
        print(f"   📁 Basel AML data: {BASEL_FILE}")
        print(f"   📁 FSRB data: {FSRB_FILE}")