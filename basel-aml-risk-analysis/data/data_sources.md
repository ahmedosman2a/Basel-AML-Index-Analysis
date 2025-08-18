# Data Sources

## Primary Data Sources

### Basel AML Index Expert Edition
- **Source**: Basel Institute on Governance
- **URL**: https://index.baselgovernance.org/downloads
- **Coverage**: 203 countries and territories
- **Variables**: 6 risk assessment pillars (Quality of Financial Intelligence, Investigation & Prosecution, Corruption, Financial Transparency, Public Transparency, Political & Legal Framework)
- **Methodology**: Risk scores on 0-10 scale based on publicly available sources and expert assessments

### FATF Global Network Classifications
- **Source**: Financial Action Task Force (FATF)
- **URL**: https://www.fatf-gafi.org/en/countries/global-network.html
- **Coverage**: FATF-Style Regional Bodies (FSRB) membership classifications
- **Variables**: Regional regulatory body assignments including FATF, APG, CFATF, EAG, ESAAMLG, GABAC, GAFILAT, GIABA, MENAFATF, MONEYVAL
- **Purpose**: Identification of anti-money laundering regulatory frameworks by jurisdiction

## Additional Data Elements

### Sanctions and Regulatory Measures
- UN Security Council sanctions
- OFAC sanctions listings
- EU restrictive measures
- FATF grey and black list classifications
- National sanctions regimes (UK, Australia)

## Data Validation
All sources verified against official publications and cross-referenced for consistency. Missing values handled through multiple imputation where appropriate.

## Last Updated
Data current as of analysis date. Refer to original sources for most recent updates.

# Data Documentation

## Dataset Overview

- **Countries**: 203
- **Variables**: 42 total (6 risk pillars + 11 sanctions + 11 FSRB + composites)
- **Source**: Basel AML Index Expert Edition + FATF Global Network
- **Processing**: Missing data imputed, robust normalization applied

## Folder Structure

- `raw/`: Original datasets as downloaded
- `processed/`: Cleaned and analysis-ready datasets
- `external/`: Reference datasets and auxiliary data

## Key Files

| File | Description | Records |
|------|-------------|---------|
| `baselamlindexexpertedition.csv` | Original Basel AML data | 203 |
| `fsrb_regions.csv` | FATF regional classifications | ~196 |
| `basel_aml_with_fsrb_complete.csv` | Final analysis dataset | 203 |

## Data Processing Steps

1. **Data Integration**: Basel AML + FSRB mapping (95% match rate)
2. **Missing Data**: MICE imputation for 10.3% missing financial transparency scores
3. **Normalization**: Robust scaling applied to all risk variables
4. **Feature Engineering**: Binary sanctions encoding, composite measures created

## Variable Categories

- **Risk Scores**: Original (0-10 scale) and robust normalized versions
- **Sanctions**: 11 binary indicators for regulatory measures
- **Regional**: FSRB membership classifications
- **Composite**: Sanction totals, weighted indices, risk categories

## Data Quality

- **Completeness**: 83% complete cases before imputation
- **Validation**: Cross-referenced with official sources
- **Missing Patterns**: Systematic for offshore centers (Asset Recovery, Public Transparency)