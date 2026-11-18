#!/usr/bin/env python3
# ==============================================================================
# Statistical Replication Script: Inter-Rater Reliability & Comparative Analysis
# Manuscript: "Bridging the Gap Between Wearable Sensors and Clinical Practice in
#             Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of
#             a Multi-Stage Large Language Model Pipeline" (JMIR Formative Research)
# Author: Ugonna Oleh et al. (November 2026 Revision)
#
# Resolves Reviewer AS Critique (R1-M1 & R1-M4):
# 1. Reproduces exact closed-form Gwet's AC2 with Linear Ordinal Weights matching
#    the official R library 'irrCAC::gwet.ac1.raw(..., weights="linear", categ.labels=1:5)'.
# 2. Employs Student's t-distribution with df = n - 1 = 2 (t_crit = 4.303 for 95% CI)
#    across the 3 rated landmark weeks (Weeks 1, 18, 23), documenting clinical discordance.
# 3. Computes Paired Differences and Wilcoxon Signed-Rank tests (Short vs. Long).
# 4. Computes Format Preference frequencies and Fisher's Exact test for order bias.
# ==============================================================================

import os
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import stats

def gwet_ac1_raw_exact(ratings_df, weights='linear', categ_labels=list(range(1, 6)), conflev=0.95, N=np.inf):
    """
    Exact mathematical replication of irrCAC::gwet.ac1.raw for ordinal ratings.
    
    Parameters:
    -----------
    ratings_df : pd.DataFrame
        Subject (rows) by Rater (columns) ratings matrix.
    weights : str
        'linear', 'quadratic', or 'unweighted'
    categ_labels : list
        Categories present in the rating scale (default: [1, 2, 3, 4, 5])
    conflev : float
        Confidence level (default: 0.95)
    N : float
        Population size for finite population correction (default: infinity)
        
    Returns:
    --------
    dict with pa, pe, ac2, se, lcb, ucb, p_val, t_crit
    """
    ratings_mat = ratings_df.values.astype(float)
    n, r = ratings_mat.shape
    f = n / N
    
    categ = np.sort(categ_labels)
    q = len(categ)
    
    # Linear weight matrix w_kl = 1 - |k - l| / (max - min)
    weights_mat = np.zeros((q, q))
    xmin = min(categ)
    xmax = max(categ)
    for k in range(q):
        for l in range(q):
            if weights == 'linear':
                weights_mat[k, l] = 1.0 - abs(categ[k] - categ[l]) / (xmax - xmin)
            elif weights == 'quadratic':
                weights_mat[k, l] = 1.0 - ((categ[k] - categ[l])**2) / ((xmax - xmin)**2)
            elif weights == 'unweighted':
                weights_mat[k, l] = 1.0 if k == l else 0.0
                
    # Agree matrix: count of raters choosing category k for subject i
    agree_mat = np.zeros((n, q))
    for k in range(q):
        agree_mat[:, k] = np.sum(ratings_mat == categ[k], axis=1)
        
    agree_mat_w = agree_mat @ weights_mat
    ri_vec = np.sum(agree_mat, axis=1)
    sum_q = np.sum(agree_mat * (agree_mat_w - 1.0), axis=1)
    
    valid_subjects = ri_vec >= 2
    n2more = np.sum(valid_subjects)
    if n2more == 0:
        return {'pa': np.nan, 'pe': np.nan, 'ac2': np.nan, 'se': np.nan,
                'lcb': np.nan, 'ucb': np.nan, 'p_val': np.nan, 't_crit': np.nan}
    
    # Observed agreement Pa
    pa = np.sum(sum_q[valid_subjects] / (ri_vec[valid_subjects] * (ri_vec[valid_subjects] - 1.0))) / n2more
    
    # Category probability vector pi
    prop_mat = agree_mat / ri_vec[:, None]
    pi_vec = np.mean(prop_mat[valid_subjects], axis=0)
    
    # Chance agreement Pe
    tw = np.sum(weights_mat)
    if q >= 2:
        pe = (tw / (q * (q - 1))) * np.sum(pi_vec * (1.0 - pi_vec))
    else:
        pe = 1e-15
        
    gwet_ac1 = (pa - pe) / (1.0 - pe)
    
    # Variance and standard error per Gwet closed-form
    den_ivec = ri_vec * (ri_vec - 1.0)
    den_ivec[~valid_subjects] = -1.0
    pa_ivec = sum_q / den_ivec
    
    pe_r2 = pe * (valid_subjects)
    ac1_ivec = (n / n2more) * (pa_ivec - pe_r2) / (1.0 - pe)
    
    pe_ivec = (tw / (q * (q - 1))) * (agree_mat @ (1.0 - pi_vec)) / ri_vec
    ac1_ivec_x = ac1_ivec - 2.0 * (1.0 - gwet_ac1) * (pe_ivec - pe) / (1.0 - pe)
    
    var_ac1 = ((1.0 - f) / (n * (n - 1))) * np.sum((ac1_ivec_x[valid_subjects] - gwet_ac1)**2) if n >= 2 else np.nan
    stderr = np.sqrt(max(0, var_ac1))
    
    # Exact Student's t critical value for df = n - 1 = 2
    df = n - 1
    t_crit = stats.t.ppf(1.0 - (1.0 - conflev) / 2.0, df=df) if n >= 2 else 1.96
    lcb = max(-1.0, gwet_ac1 - stderr * t_crit)
    ucb = min(1.0, gwet_ac1 + stderr * t_crit)
    p_val = 1.0 - stats.t.cdf(gwet_ac1 / stderr, df=df) if (stderr > 0 and not np.isnan(stderr)) else np.nan
    
    return {
        'pa': pa, 'pe': pe, 'ac2': gwet_ac1, 'se': stderr,
        'lcb': lcb, 'ucb': ucb, 'p_val': p_val, 't_crit': t_crit
    }

def main():
    parser = argparse.ArgumentParser(description="Statistical Replication for JMIR Formative Research ms#105019")
    parser.add_argument('-i', '--input', default='raw_expert_ratings.csv', help='Path to raw expert ratings CSV')
    parser.add_argument('-p', '--preferences', default='format_preferences.csv', help='Path to format preferences CSV')
    parser.add_argument('-o', '--output_dir', default='.', help='Output directory for generated CSVs')
    args = parser.parse_args()

    print("=" * 80)
    print("CLINICIAN INTER-RATER RELIABILITY & COMPARATIVE EVALUATION REPLICATION")
    print("Manuscript: ms#105019 (JMIR Formative Research)")
    print(f"Ratings File:     {args.input}")
    print(f"Preferences File: {args.preferences}")
    print("=" * 80)

    if not os.path.exists(args.input):
        print(f"Error: Ratings file not found at {args.input}", file=sys.stderr)
        sys.exit(1)

    raw_df = pd.read_csv(args.input)

    # Normalize column names and format values for maximum compatibility across CSV schemas
    col_mapping = {
        'Clinician_ID': 'rater_id',
        'Format': 'report_format',
        'Week': 'week',
        'Information_Hierarchy_A1': 'A1_information_hierarchy',
        'Conciseness_A2': 'A2_conciseness',
        'Context_Factors_B3': 'B3_identification_of_factors',
        'Risk_Plausibility_B4': 'B4_likelihood_assessment',
        'Clinical_Relevance_C5': 'C5_relevance',
        'Appropriateness_C6': 'C6_appropriateness',
        'Session_Prep_Value_D7': 'D7_overall_value_session_prep'
    }
    raw_df = raw_df.rename(columns=col_mapping)
    if 'report_format' in raw_df.columns:
        raw_df['report_format'] = raw_df['report_format'].replace({
            'Long-Form': 'Long_Form_Research_Log',
            'Short-Form': 'Short_Form_Psychologist_Report'
        })

    metrics_map = [
        ('A1_information_hierarchy', 'Information Hierarchy (A.1)'),
        ('A2_conciseness', 'Conciseness (A.2)'),
        ('B3_identification_of_factors', 'Identification of Context Factors (B.3)'),
        ('B4_likelihood_assessment', 'Perceived Plausibility of Risk Assessment (B.4)'),
        ('C5_relevance', 'Clinical Relevance (C.5)'),
        ('C6_appropriateness', 'Actionability & Appropriateness (C.6)'),
        ('D7_overall_value_session_prep', 'Overall Value for Session Prep (D.7)')
    ]

    formats = [
        ('Long_Form_Research_Log', 'Long-Form (Research Log)'),
        ('Short_Form_Psychologist_Report', 'Short-Form (Psychologist Report)')
    ]

    # 1. TABLE 4 RECONSTRUCTION: SUMMARY STATS & GWET'S AC2
    print("\n" + "-" * 80)
    print("TABLE 4: DESCRIPTIVE STATISTICS & REPRODUCIBLE GWET'S AC2 (N=7 Raters, 3 Weeks)")
    print("-" * 80)

    results_summary = []

    for fmt_key, fmt_label in formats:
        print(f"\n>>> FORMAT: {fmt_label} <<<")
        sub_df = raw_df[raw_df['report_format'] == fmt_key]
        
        for col_key, col_label in metrics_map:
            vals = sub_df[col_key].dropna().values.astype(float)
            n_obs = len(vals)
            n_missing = 21 - n_obs
            med = np.median(vals)
            q1 = np.percentile(vals, 25)
            q3 = np.percentile(vals, 75)
            iqr = q3 - q1
            mean_val = np.mean(vals)
            sd_val = np.std(vals, ddof=1)
            
            # Gwet AC2 matrix: weeks (items=3) x raters (7)
            mat = sub_df.pivot(index='week', columns='rater_id', values=col_key)
            ac2_res = gwet_ac1_raw_exact(mat, weights='linear', categ_labels=list(range(1, 6)))
            
            print(f"{col_label:<42} | Med [IQR]: {med:.1f} [{iqr:.1f}] | Mean (SD): {mean_val:.2f} ({sd_val:.2f}) | AC2: {ac2_res['ac2']:6.4f} [{ac2_res['lcb']:6.3f}, {ac2_res['ucb']:6.3f}]")
            
            results_summary.append({
                'Format': fmt_label,
                'Criterion': col_label,
                'N_obs': n_obs,
                'N_missing': n_missing,
                'Median': med,
                'Q1': q1,
                'Q3': q3,
                'IQR': iqr,
                'Mean': round(mean_val, 2),
                'SD': round(sd_val, 2),
                'Pa': round(ac2_res['pa'], 4),
                'Pe': round(ac2_res['pe'], 4),
                'AC2': round(ac2_res['ac2'], 4),
                'SE': round(ac2_res['se'], 4),
                'CI95_Lower': round(ac2_res['lcb'], 3),
                'CI95_Upper': round(ac2_res['ucb'], 3),
                'p_value': round(ac2_res['p_val'], 4) if pd.notna(ac2_res['p_val']) else np.nan
            })

    df_res = pd.DataFrame(results_summary)
    out_table4 = os.path.join(args.output_dir, 'table4_gwet_ac2_reproduced.csv')
    df_res.to_csv(out_table4, index=False)
    print(f"\nSaved Table 4 reproduction to: {out_table4}")

    # 2. PAIRED DIFFERENCES ANALYSIS (SHORT-FORM MINUS LONG-FORM)
    print("\n" + "-" * 80)
    print("PAIRED DIFFERENCES & WILCOXON SIGNED-RANK TESTS (Short minus Long)")
    print("-" * 80)

    paired_records = []
    for col_key, col_label in metrics_map:
        diffs = []
        for rater in raw_df['rater_id'].unique():
            for wk in raw_df['week'].unique():
                row_l = raw_df[(raw_df['rater_id'] == rater) & (raw_df['week'] == wk) & (raw_df['report_format'] == 'Long_Form_Research_Log')]
                row_s = raw_df[(raw_df['rater_id'] == rater) & (raw_df['week'] == wk) & (raw_df['report_format'] == 'Short_Form_Psychologist_Report')]
                if len(row_l) > 0 and len(row_s) > 0:
                    vl = row_l[col_key].values[0]
                    vs = row_s[col_key].values[0]
                    if pd.notna(vl) and pd.notna(vs):
                        diffs.append(float(vs) - float(vl))
                        
        d = np.array(diffs)
        n_pairs = len(d)
        med_d = np.median(d)
        q1_d = np.percentile(d, 25)
        q3_d = np.percentile(d, 75)
        iqr_d = q3_d - q1_d
        mean_d = np.mean(d)
        sd_d = np.std(d, ddof=1)
        
        if np.all(d == 0):
            w_stat, p_val = 0.0, 1.0
        else:
            try:
                w_res = stats.wilcoxon(d, alternative='two-sided')
                w_stat, p_val = float(w_res.statistic), float(w_res.pvalue)
            except Exception:
                w_stat, p_val = np.nan, np.nan
                
        print(f"{col_label:<42} | N: {n_pairs:2d} | Med Diff: {med_d:+.1f} [IQR: {iqr_d:.1f}] | Mean (SD): {mean_d:+.2f} ({sd_d:.2f}) | W = {w_stat:4.1f}, p = {p_val:.4f}")
        
        paired_records.append({
            'Criterion': col_label,
            'N_pairs': n_pairs,
            'Median_Diff': med_d,
            'Q1_Diff': q1_d,
            'Q3_Diff': q3_d,
            'IQR_Diff': iqr_d,
            'Mean_Diff': round(mean_d, 2),
            'SD_Diff': round(sd_d, 2),
            'Wilcoxon_W': w_stat,
            'p_value': round(p_val, 4)
        })

    df_paired = pd.DataFrame(paired_records)
    out_paired = os.path.join(args.output_dir, 'paired_differences_reproduced.csv')
    df_paired.to_csv(out_paired, index=False)
    print(f"\nSaved Paired Differences reproduction to: {out_paired}")

    # 3. FORCED-CHOICE FORMAT PREFERENCE & ORDER BIAS ANALYSIS
    if os.path.exists(args.preferences):
        print("\n" + "-" * 80)
        print("PART 4: FORCED-CHOICE FORMAT PREFERENCE & PRESENTATION ORDER BIAS")
        print("-" * 80)
        
        try:
            pref_df = pd.read_csv(args.preferences, on_bad_lines='skip')
        except Exception:
            import csv
            rows = []
            with open(args.preferences, 'r', encoding='utf-8', errors='ignore') as f:
                reader = csv.reader(f)
                header = next(reader)
                for r in reader:
                    if len(r) >= 2:
                        rows.append({'Order': r[0], 'Preference': r[1]})
            pref_df = pd.DataFrame(rows)
        n_evals = len(pref_df)
        
        # Categorize choices
        pref_df['Choice'] = pref_df['Preference'].apply(
            lambda x: 'Short-Form' if 'Short' in str(x) else ('Long-Form' if 'Long' in str(x) else 'Other')
        )
        
        short_count = (pref_df['Choice'] == 'Short-Form').sum()
        long_count = (pref_df['Choice'] == 'Long-Form').sum()
        
        print(f"Total Completed Format Evaluations: N = {n_evals}")
        print(f"  - Preferred Short-Form (Psychologist Report): {short_count} ({short_count / n_evals * 100:.1f}%)")
        print(f"  - Preferred Long-Form (Research Log):         {long_count} ({long_count / n_evals * 100:.1f}%)")
        
        # Contingency table
        ct = pd.crosstab(pref_df['Order'], pref_df['Choice'])
        print("\nContingency Table (Presentation Order vs Format Preference):")
        print(ct)
        
        # Fisher's Exact Test
        if ct.shape == (2, 2):
            odds_ratio, p_fisher = stats.fisher_exact(ct.values)
            print(f"\nFisher's Exact Test for Order Bias: Odds Ratio = {odds_ratio:.3f}, p = {p_fisher:.3f}")
            if p_fisher > 0.05:
                print("Conclusion: No statistically significant presentation order bias detected (p > .05).")
            else:
                print("Conclusion: Presentation order significantly influenced format choice.")

    print("\n" + "=" * 80)
    print("REPLICATION COMPLETED SUCCESSFULLY")
    print("=" * 80)

if __name__ == '__main__':
    main()
