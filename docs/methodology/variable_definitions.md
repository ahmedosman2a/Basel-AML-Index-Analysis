# Variable Definitions

## Basel AML Risk Variables

### Original Risk Scores (0-10 scale)
| Variable | Description | Range |
|----------|-------------|-------|
| `score` | Overall AML risk score | 2.96 - 8.20 |
| `mltf` | Money Laundering/Terrorist Financing | 2.57 - 9.07 |
| `cor` | Corruption risk assessment | 1.85 - 8.68 |
| `fintran` | Financial transparency measures | 0.07 - 10.00 |
| `pubtran` | Public transparency indicators | 0.00 - 10.00 |
| `polleg` | Political & legal framework | 0.29 - 9.43 |

### Robust Normalized Variables
| Variable | Description | Properties |
|----------|-------------|------------|
| `score_robust` | Robust scaled overall score | Median=0, IQR-based |
| `mltf_robust` | Robust scaled MLTF | Outlier-resistant |
| `cor_robust` | Robust scaled corruption | Better for modeling |
| `fintran_robust` | Robust scaled financial transparency | Handles zeros |
| `pubtran_robust` | Robust scaled public transparency | Wide range accommodation |
| `polleg_robust` | Robust scaled political/legal | Statistical optimization |

## Sanctions and Regulatory Variables

### Binary Sanction Indicators (0/1)
| Variable | Description | Coverage |
|----------|-------------|----------|
| `ofac_binary` | US OFAC sanctions | 29 countries (14.2%) |
| `un_sc_binary` | UN Security Council sanctions | 14 countries (6.9%) |
| `eu_restrictive_measures_binary` | EU restrictive measures | 28 countries (13.7%) |
| `eu_high_risk_binary` | EU high-risk third countries | 26 countries (12.7%) |
| `fatf_grey_binary` | FATF increased monitoring | 24 countries (11.8%) |
| `fatf_black_binary` | FATF call for action | 3 countries (1.5%) |
| `uk_sc_binary` | UK sanctions | 27 countries (13.2%) |
| `uk_fs_binary` | UK financial sanctions | 23 countries (11.3%) |
| `australian_sc_binary` | Australian sanctions | 7 countries (3.4%) |
| `eu_non_cooperative_tax_binary` | EU tax non-cooperation | 8 countries (3.9%) |
| `eu_non_cooperative_taxc_binary` | EU tax cooperation list | 8 countries (3.9%) |

### Composite Sanction Measures
| Variable | Description | Calculation |
|----------|-------------|-------------|
| `sanction_total` | Total number of sanctions | Sum of binary indicators |
| `sanction_weighted` | Weighted sanction index | UN(1.0) + FATF(0.9) + Major(0.8) + Others(0.6) |
| `sanction_level` | Categorical sanction burden | None/Low/Medium/High |

## Regional Classifications

### FSRB Variables
| Variable | Description | Member Countries |
|----------|-------------|------------------|
| `primary_fsrb` | Primary FSRB membership | Main regional classification |
| `fsrb_fatf` | FATF membership | Direct FATF members |
| `fsrb_apg` | Asia/Pacific Group | APG region |
| `fsrb_moneyval` | Council of Europe | MONEYVAL region |
| `fsrb_cfatf` | Caribbean FATF | Caribbean region |
| `fsrb_esaamlg` | Eastern/Southern Africa | ESAAMLG region |
| `fsrb_menafatf` | Middle East/North Africa | MENAFATF region |
| `fsrb_gafilat` | Latin America | GAFILAT region |
| `fsrb_giaba` | West Africa | GIABA region |
| `fsrb_eag` | Eurasian Group | EAG region |
| `fsrb_gabac` | Central Africa | GABAC region |

### Additional Variables
| Variable | Description | Values |
|----------|-------------|---------|
| `fsrb_count` | Number of FSRB memberships | 0-2 (some countries have multiple) |
| `has_fsrb_data` | Data availability indicator | 1=matched, 0=unmatched |

## Data Types and Scales

- **Continuous**: All risk scores (original and robust)
- **Binary**: All sanction and FSRB indicators
- **Ordinal**: Sanction level categories
- **Categorical**: Primary FSRB, country names
- **Count**: Sanction total, FSRB count