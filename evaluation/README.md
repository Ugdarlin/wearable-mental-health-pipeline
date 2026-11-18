# Clinical Evaluation & Statistical Replication Package

**Manuscript:** *Bridging the Gap Between Wearable Sensors and Clinical Practice in Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of a Multi-Stage Large Language Model Pipeline*  
**Journal:** *JMIR Formative Research* (Manuscript ID: **ms#105019**)  
**Authors:** Ugonna Oleh, Alla Machulska, Tim Klucken, Roman Obermaisser  
**Revision:** November 2026  

---

## Directory Overview

This directory contains the complete data, evaluation instruments, and reproducible statistical analysis scripts for the clinician evaluation reported in the manuscript:

| File Name | Description |
| :--- | :--- |
| [`raw_expert_ratings.csv`](raw_expert_ratings.csv) | Full item-level ratings matrix from $N=7$ licensed/practicing clinicians across 3 landmark evaluation weeks (Weeks 1, 18, 23) and 2 report formats (Long-Form Research Log vs. Short-Form Psychologist Report), covering 7 evaluation criteria and qualitative feedback (40 completed evaluations, 42 rows). |
| [`format_preferences.csv`](format_preferences.csv) | Part 4 forced-choice format preferences, randomized presentation order (`Long-Short` vs. `Short-Long`), clinician rationales, and qualitative commentary across the 20 completed format evaluations. |
| [`clinical_questionnaire.pdf`](clinical_questionnaire.pdf) | The formal 4-part evaluation instrument administered to clinicians (Likert rating scales, qualitative prompts, and forced-choice format comparison). |
| [`gwet_ac2_analysis.R`](gwet_ac2_analysis.R) | **Official R Replication Script**: Computes Gwet's $\text{AC}_2$ inter-rater reliability using the established peer-reviewed [`irrCAC`](https://cran.r-project.org/package=irrCAC) package (`irrCAC::gwet.ac1.raw`) with linear ordinal weights, Student's $t$ confidence intervals ($df=2$), paired Wilcoxon signed-rank tests, and Fisher's exact test. |
| [`statistical_analysis.py`](statistical_analysis.py) | **Python Replication Script**: Self-contained Python script implementing the exact closed-form Gwet $\text{AC}_2$ benchmark matching `irrCAC`, generating identical descriptive statistics, paired difference tests, and preference analysis. |

---

## Key Methodological & Statistical Specifications (Addressing Peer Review)

In response to peer review (Reviewer AS, Critiques R1-M1 and R1-M4), this repository adheres strictly to verified statistical standards:

1. **Standard `irrCAC` Implementation:**
   - Previous informal scripts implemented custom agreement functions that diverged from published formulations. The replication scripts now directly utilize or mathematically mirror `irrCAC::gwet.ac1.raw(..., weights = "linear", categ.labels = 1:5)`.
   - Linear ordinal weighting is defined as:
     $$w_{kl} = 1 - \frac{|k - l|}{5 - 1} = 1 - \frac{|k - l|}{4}$$
     for categories $k, l \in \{1, 2, 3, 4, 5\}$.

2. **Degrees of Freedom and Confidence Intervals ($df = 2$):**
   - Because the unit of statistical rating is the landmark calendar week ($N = 3$ rated weeks: Week 1, Week 18, and Week 23), the degrees of freedom for the jackknife standard error are:
     $$df = N - 1 = 3 - 1 = 2$$
   - The two-sided 95% confidence interval requires the Student's $t$ critical value:
     $$t_{0.975, \, 2} = 4.30265 \approx 4.303$$
   - This accounts for why confidence intervals are wide (e.g., Long-Form Information Hierarchy $\text{AC}_2 = 0.668$, 95% CI 0.462 to 0.874; Short-Form Conciseness $\text{AC}_2 = 0.278$, 95% CI -0.115 to 0.670). As discussed in the manuscript, rather than masking this variance with an unjustified normal $z=1.96$ approximation, the wide intervals are transparently reported as empirical evidence of genuine clinical subjectivity and rater discordance among mental health professionals.

3. **Paired Differences (Short-Form vs. Long-Form):**
   - Clinician ratings are paired at the individual evaluator and evaluation week level ($N = 19\text{--}20$ pairs per criterion).
   - Paired differences are evaluated via two-sided Wilcoxon signed-rank tests with continuity correction.

4. **Forced-Choice Preference & Order Bias:**
   - Evaluated via a $2 \times 2$ contingency table cross-tabulating presentation order (`Long-Short` vs. `Short-Long`) against format preference (`Short-Form` vs. `Long-Form`), analyzed with Fisher's exact test ($P = .603$, two-sided).

---

## How to Run the Replication

### Option A: Running via R (Official Reference)

**Prerequisites:** R version 4.0 or higher.
Install the required packages in R:
```R
install.packages(c("irrCAC", "tidyverse", "optparse"), repos = "https://cloud.r-project.org")
```

**Execution:**
```bash
# Run from the evaluation directory:
Rscript gwet_ac2_analysis.R --input raw_expert_ratings.csv --preferences format_preferences.csv
```

### Option B: Running via Python

**Prerequisites:** Python 3.9+ with standard scientific stack:
```bash
pip install numpy pandas scipy
```

**Execution:**
```bash
# Run from the evaluation directory:
python statistical_analysis.py --input raw_expert_ratings.csv --preferences format_preferences.csv
```

Both scripts generate:
- `table4_gwet_ac2_reproduced.csv` (Matches Table 4 in manuscript v12)
- `paired_differences_reproduced.csv` (Matches paired statistical results in manuscript v12 and Multimedia Appendix 3)

---

## Expected Benchmark Results (Table 4 Reproduction)

### 1. Long-Form (Research Log)

| Criterion | $N_{obs}$ | Median [IQR] | Mean (SD) | $P_a$ | $P_e$ | Gwet's $\text{AC}_2$ | SE | 95% CI ($df=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A.1 Information Hierarchy** | 20 | 3.0 [1.0] | 3.05 (0.69) | 0.8190 | 0.4549 | **0.6680** | 0.0479 | [0.462, 0.874] |
| **A.2 Conciseness** | 20 | 2.5 [1.0] | 2.65 (0.67) | 0.7714 | 0.4693 | **0.5694** | 0.0381 | [0.406, 0.733] |
| **B.3 Context Factors** | 20 | 2.5 [1.2] | 2.75 (0.79) | 0.7381 | 0.4852 | **0.4912** | 0.0763 | [0.163, 0.820] |
| **B.4 Risk Assessment Plausibility** | 20 | 2.0 [1.0] | 2.05 (0.69) | 0.7810 | 0.4800 | **0.5787** | 0.0637 | [0.305, 0.853] |
| **C.5 Clinical Relevance** | 20 | 2.5 [1.2] | 2.75 (0.85) | 0.7095 | 0.4729 | **0.4489** | 0.0522 | [0.224, 0.674] |
| **C.6 Actionability & Appropriateness** | 20 | 3.0 [1.2] | 2.85 (0.88) | 0.6857 | 0.4716 | **0.4052** | 0.0468 | [0.204, 0.606] |
| **D.7 Overall Session Prep Value** | 19 | 2.0 [1.0] | 2.16 (0.60) | 0.7963 | 0.4804 | **0.6080** | 0.0249 | [0.501, 0.715] |

### 2. Short-Form (Psychologist Report)

| Criterion | $N_{obs}$ | Median [IQR] | Mean (SD) | $P_a$ | $P_e$ | Gwet's $\text{AC}_2$ | SE | 95% CI ($df=2$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **A.1 Information Hierarchy** | 20 | 3.0 [1.2] | 3.10 (0.91) | 0.6905 | 0.4789 | **0.4060** | 0.0935 | [0.004, 0.808] |
| **A.2 Conciseness** | 20 | 3.0 [1.0] | 3.20 (1.06) | 0.6143 | 0.4661 | **0.2775** | 0.0912 | [-0.115, 0.670] |
| **B.3 Context Factors** | 20 | 2.0 [1.0] | 2.45 (0.89) | 0.7143 | 0.4905 | **0.4393** | 0.0253 | [0.330, 0.548] |
| **B.4 Risk Assessment Plausibility** | 19 | 3.0 [1.5] | 2.89 (0.88) | 0.6865 | 0.4815 | **0.3955** | 0.0617 | [0.130, 0.661] |
| **C.5 Clinical Relevance** | 20 | 2.0 [1.2] | 2.30 (0.98) | 0.6381 | 0.4788 | **0.3056** | 0.0760 | [-0.021, 0.632] |
| **C.6 Actionability & Appropriateness** | 20 | 2.0 [1.0] | 2.40 (0.88) | 0.6857 | 0.4839 | **0.3910** | 0.0537 | [0.160, 0.622] |
| **D.7 Overall Session Prep Value** | 20 | 2.0 [1.0] | 2.00 (0.86) | 0.6905 | 0.4862 | **0.3976** | 0.0469 | [0.196, 0.599] |

### 3. Paired Differences Analysis (Short minus Long, Wilcoxon Signed-Rank Tests)

| Criterion | $N_{pairs}$ | Median Diff [IQR] | Mean Diff (SD) | Wilcoxon $W$ | $p$-value |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **A.1 Information Hierarchy** | 20 | 0.0 [1.0] | +0.05 (0.83) | 27.5 | .838 |
| **A.2 Conciseness** | 20 | +1.0 [1.2] | +0.55 (0.76) | 10.5 | **.003** |
| **B.3 Context Factors** | 20 | 0.0 [1.0] | -0.30 (0.86) | 33.0 | .194 |
| **B.4 Risk Assessment Plausibility** | 19 | 0.0 [1.5] | +0.89 (1.10) | 18.0 | .018 |
| **C.5 Clinical Relevance** | 20 | 0.0 [1.0] | -0.45 (0.83) | 22.0 | .061 |
| **C.6 Actionability & Appropriateness** | 20 | 0.0 [1.2] | -0.45 (0.94) | 26.0 | .088 |
| **D.7 Overall Session Prep Value** | 19 | 0.0 [1.0] | -0.16 (0.83) | 21.0 | .481 |

*Note: Across 6 of 7 criteria, median paired differences were exactly 0.0, indicating no systematic preference for Short-Form over Long-Form outside of Conciseness (A.2).*

---

## Citation & Contact

If using or referencing these evaluation data or scripts, please cite:
> Oleh U, Machulska A, Klucken T, Obermaisser R. Bridging the Gap Between Wearable Sensors and Clinical Practice in Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of a Multi-Stage Large Language Model Pipeline. *JMIR Formative Research*, ms#105019, 2026.
