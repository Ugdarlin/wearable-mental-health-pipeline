# ==============================================================================
# Statistical Replication Script: Inter-Rater Reliability & Comparative Analysis
# Manuscript: "Bridging the Gap Between Wearable Sensors and Clinical Practice in
#             Agoraphobia and Panic Disorder: Single-Case Formative Evaluation of
#             a Multi-Stage Large Language Model Pipeline" (JMIR Formative Research)
# Author: Ugonna Oleh et al. (November 2026 Revision)
#
# Resolves Reviewer AS Critique (R1-M1 & R1-M4):
# 1. Utilizes standard, peer-reviewed 'irrCAC' library (gwet.ac1.raw) with Linear
#    Ordinal Weights for 5-point Likert categories (1 to 5).
# 2. Computes exact Student's t-distribution confidence intervals with df = n - 1 = 2
#    (t_crit = 4.303 for 95% CI across 3 landmark weeks), reflecting genuine rater discordance.
# 3. Executes Paired Wilcoxon Signed-Rank tests (Short-Form vs. Long-Form).
# 4. Executes Fisher's Exact Test for format preference presentation order bias.
# ==============================================================================

# Ensure required packages are installed
required_packages <- c("optparse", "tidyverse")
for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org", quietly = TRUE)
  }
}
suppressPackageStartupMessages({
  library(optparse)
  library(tidyverse)
})

# Attempt to load irrCAC; if not available, install it
has_irrCAC <- requireNamespace("irrCAC", quietly = TRUE)
if (!has_irrCAC) {
  tryCatch({
    install.packages("irrCAC", repos = "https://cloud.r-project.org", quietly = TRUE)
    has_irrCAC <- requireNamespace("irrCAC", quietly = TRUE)
  }, error = function(e) {
    message("Notice: irrCAC package could not be downloaded automatically from CRAN.")
  })
}

# ------------------------------------------------------------------------------
# Closed-Form Reference Implementation of Gwet's AC2 (Matching irrCAC::gwet.ac1.raw)
# Used as guaranteed fallback if internet access prevents dynamic package install
# ------------------------------------------------------------------------------
calc_gwet_ac2_reference <- function(ratings_mat, q = 5, conflev = 0.95) {
  # ratings_mat: n_items (weeks=3) x r_raters (7)
  n <- nrow(ratings_mat)
  r <- ncol(ratings_mat)
  
  # Linear weight matrix: w_kl = 1 - |k - l| / (q - 1)
  w <- matrix(0, nrow = q, ncol = q)
  for (k in 1:q) {
    for (l in 1:q) {
      w[k, l] <- 1.0 - abs(k - l) / (q - 1)
    }
  }
  
  # Agreement counts matrix: r_ik = count of raters rating category k on item i
  agree_mat <- matrix(0, nrow = n, ncol = q)
  for (i in 1:n) {
    for (k in 1:q) {
      agree_mat[i, k] <- sum(ratings_mat[i, ] == k, na.rm = TRUE)
    }
  }
  
  ri_vec <- rowSums(agree_mat)
  valid <- ri_vec >= 2
  n2more <- sum(valid)
  if (n2more == 0) return(list(pa = NA, pe = NA, ac2 = NA, se = NA, ci_lower = NA, ci_upper = NA, p_val = NA))
  
  agree_mat_w <- agree_mat %*% w
  sum_q <- rowSums(agree_mat * (agree_mat_w - 1.0))
  
  # Observed agreement Pa
  pa <- sum(sum_q[valid] / (ri_vec[valid] * (ri_vec[valid] - 1.0))) / n2more
  
  # Category probability vector pi
  prop_mat <- agree_mat / ri_vec
  pi_vec <- colMeans(prop_mat[valid, , drop = FALSE])
  
  # Chance agreement Pe
  tw <- sum(w)
  pe <- (tw / (q * (q - 1))) * sum(pi_vec * (1.0 - pi_vec))
  
  # Gwet's AC2
  ac2 <- (pa - pe) / (1.0 - pe)
  
  # Variance & Standard Error (Gwet closed-form)
  den_i <- ri_vec * (ri_vec - 1.0)
  den_i[!valid] <- -1.0
  pa_i <- sum_q / den_i
  ac1_i <- (n / n2more) * (pa_i - pe) / (1.0 - pe)
  
  pe_i <- (tw / (q * (q - 1))) * as.vector(agree_mat %*% (1.0 - pi_vec)) / ri_vec
  ac1_i_x <- ac1_i - 2.0 * (1.0 - ac2) * (pe_i - pe) / (1.0 - pe)
  
  var_ac2 <- (1.0 / (n * (n - 1))) * sum((ac1_i_x[valid] - ac2)^2)
  se_ac2 <- sqrt(max(0, var_ac2))
  
  # Exact Student's t critical value for df = n - 1 = 2
  df <- n - 1
  t_crit <- qt(1.0 - (1.0 - conflev) / 2.0, df = df)
  
  ci_lower <- max(-1.0, ac2 - t_crit * se_ac2)
  ci_upper <- min(1.0, ac2 + t_crit * se_ac2)
  p_val <- 1.0 - pt(ac2 / se_ac2, df = df)
  
  return(list(pa = pa, pe = pe, ac2 = ac2, se = se_ac2, ci_lower = ci_lower, ci_upper = ci_upper, p_val = p_val, t_crit = t_crit))
}

# ------------------------------------------------------------------------------
# Dispatcher for Gwet's AC2
# ------------------------------------------------------------------------------
run_gwet_ac2 <- function(ratings_df, q = 5, conflev = 0.95) {
  if (has_irrCAC) {
    tryCatch({
      res <- irrCAC::gwet.ac1.raw(ratings_df, weights = "linear", categ.labels = 1:q, conflev = conflev)
      est <- res$est
      return(list(
        pa = est["pa", "coeff.val"],
        pe = est["pe", "coeff.val"],
        ac2 = est["gwet.ac1", "coeff.val"],
        se = est["gwet.ac1", "coeff.se"],
        ci_lower = est["gwet.ac1", "conf.int"] %>% strsplit(",") %>% { as.numeric(gsub("\\[|\\(| ", "", .[[1]][1])) },
        ci_upper = est["gwet.ac1", "conf.int"] %>% strsplit(",") %>% { as.numeric(gsub("\\]|\\)| ", "", .[[1]][2])) },
        p_val = est["gwet.ac1", "p.value"],
        method = "irrCAC::gwet.ac1.raw (Linear Weights)"
      ))
    }, error = function(e) {
      # Fallback to reference math
    })
  }
  
  ref <- calc_gwet_ac2_reference(as.matrix(ratings_df), q = q, conflev = conflev)
  ref$method <- "Exact Gwet Reference (Linear Weights, df=2)"
  return(ref)
}

# ------------------------------------------------------------------------------
# Main Analysis Execution
# ------------------------------------------------------------------------------
option_list <- list(
  make_option(c("-i", "--input"), type = "character", default = "raw_expert_ratings.csv", help = "Input ratings CSV file"),
  make_option(c("-p", "--preferences"), type = "character", default = "Study Results - Reports_Comparison.csv", help = "Input preferences CSV file"),
  make_option(c("-o", "--output_dir"), type = "character", default = ".", help = "Directory to save output CSVs")
)
opt <- parse_args(OptionParser(option_list = option_list))

cat("==============================================================================\n")
cat("CLINICIAN INTER-RATER RELIABILITY & COMPARATIVE EVALUATION REPLICATION\n")
cat("Manuscript: ms#105019 (JMIR Formative Research)\n")
cat("Ratings File:      ", opt$input, "\n")
cat("Preferences File:  ", opt$preferences, "\n")
cat("==============================================================================\n\n")

if (!file.exists(opt$input)) {
  stop(paste("Ratings file not found at:", opt$input))
}

data <- read.csv(opt$input, stringsAsFactors = FALSE)

# Normalize column names and format labels for cross-schema compatibility
col_mapping <- c(
  "Clinician_ID"             = "rater_id",
  "Format"                   = "report_format",
  "Week"                     = "week",
  "Information_Hierarchy_A1" = "A1_information_hierarchy",
  "Conciseness_A2"           = "A2_conciseness",
  "Context_Factors_B3"       = "B3_identification_of_factors",
  "Risk_Plausibility_B4"     = "B4_likelihood_assessment",
  "Clinical_Relevance_C5"    = "C5_relevance",
  "Appropriateness_C6"       = "C6_appropriateness",
  "Session_Prep_Value_D7"    = "D7_overall_value_session_prep"
)
for (orig_col in names(col_mapping)) {
  if (orig_col %in% colnames(data)) {
    colnames(data)[colnames(data) == orig_col] <- col_mapping[[orig_col]]
  }
}
if ("report_format" %in% colnames(data)) {
  data$report_format[data$report_format == "Long-Form"] <- "Long_Form_Research_Log"
  data$report_format[data$report_format == "Short-Form"] <- "Short_Form_Psychologist_Report"
}

metrics <- list(
  "A1_information_hierarchy"     = "Information Hierarchy (A.1)",
  "A2_conciseness"               = "Conciseness (A.2)",
  "B3_identification_of_factors" = "Identification of Context Factors (B.3)",
  "B4_likelihood_assessment"     = "Perceived Plausibility of Risk Assessment (B.4)",
  "C5_relevance"                 = "Clinical Relevance (C.5)",
  "C6_appropriateness"           = "Actionability & Appropriateness (C.6)",
  "D7_overall_value_session_prep"= "Overall Value for Session Prep (D.7)"
)

formats <- list(
  "Long_Form_Research_Log"         = "Long-Form (Research Log)",
  "Short_Form_Psychologist_Report" = "Short-Form (Psychologist Report)"
)

# 1. TABLE 4 RECONSTRUCTION: SUMMARY STATS & GWET'S AC2
cat("------------------------------------------------------------------------------\n")
cat("TABLE 4: DESCRIPTIVE STATISTICS & REPRODUCIBLE GWET'S AC2 (N=7 Raters, 3 Weeks)\n")
cat("------------------------------------------------------------------------------\n")

table4_results <- list()

for (fmt_key in names(formats)) {
  fmt_label <- formats[[fmt_key]]
  cat(sprintf("\n>>> FORMAT: %s <<<\n", fmt_label))
  sub_df <- data %>% filter(report_format == fmt_key)
  
  for (col_key in names(metrics)) {
    col_label <- metrics[[col_key]]
    vals <- as.numeric(sub_df[[col_key]])
    vals_clean <- vals[!is.na(vals)]
    
    n_obs <- length(vals_clean)
    n_miss <- 21 - n_obs
    med_val <- median(vals_clean)
    q1_val <- quantile(vals_clean, 0.25)
    q3_val <- quantile(vals_clean, 0.75)
    iqr_val <- q3_val - q1_val
    mean_val <- mean(vals_clean)
    sd_val <- sd(vals_clean)
    
    # Reshape into Weeks (rows, N=3) x Raters (columns, r=7)
    mat_wide <- sub_df %>%
      select(week, rater_id, all_of(col_key)) %>%
      pivot_wider(names_from = rater_id, values_from = all_of(col_key)) %>%
      select(-week)
    
    ac2_res <- run_gwet_ac2(as.data.frame(mat_wide), q = 5, conflev = 0.95)
    
    cat(sprintf("%-42s | Med [IQR]: %.1f [%.1f] | Mean (SD): %.2f (%.2f) | AC2: %6.4f [%6.3f, %6.3f]\n",
                col_label, med_val, iqr_val, mean_val, sd_val,
                ac2_res$ac2, ac2_res$ci_lower, ac2_res$ci_upper))
    
    table4_results[[length(table4_results) + 1]] <- data.frame(
      Format = fmt_label,
      Criterion = col_label,
      N_obs = n_obs,
      N_missing = n_miss,
      Median = med_val,
      Q1 = q1_val,
      Q3 = q3_val,
      IQR = iqr_val,
      Mean = round(mean_val, 2),
      SD = round(sd_val, 2),
      Pa = round(ac2_res$pa, 4),
      Pe = round(ac2_res$pe, 4),
      AC2 = round(ac2_res$ac2, 4),
      SE = round(ac2_res$se, 4),
      CI95_Lower = round(ac2_res$ci_lower, 3),
      CI95_Upper = round(ac2_res$ci_upper, 3),
      stringsAsFactors = FALSE
    )
  }
}

df_table4 <- bind_rows(table4_results)
write.csv(df_table4, file.path(opt$output_dir, "table4_gwet_ac2_reproduced.csv"), row.names = FALSE)
cat("\nSaved Table 4 reproduction to: table4_gwet_ac2_reproduced.csv\n")

# 2. PAIRED DIFFERENCES ANALYSIS (SHORT-FORM MINUS LONG-FORM)
cat("\n------------------------------------------------------------------------------\n")
cat("PAIRED DIFFERENCES & WILCOXON SIGNED-RANK TESTS (Short minus Long)\n")
cat("------------------------------------------------------------------------------\n")

paired_results <- list()

for (col_key in names(metrics)) {
  col_label <- metrics[[col_key]]
  
  # Join Long and Short by rater_id and week
  long_sub <- data %>% filter(report_format == "Long_Form_Research_Log") %>% select(rater_id, week, val_long = all_of(col_key))
  short_sub <- data %>% filter(report_format == "Short_Form_Psychologist_Report") %>% select(rater_id, week, val_short = all_of(col_key))
  
  paired_df <- inner_join(long_sub, short_sub, by = c("rater_id", "week")) %>%
    filter(!is.na(val_long) & !is.na(val_short)) %>%
    mutate(diff = val_short - val_long)
  
  diffs <- paired_df$diff
  n_pairs <- length(diffs)
  med_d <- median(diffs)
  q1_d <- quantile(diffs, 0.25)
  q3_d <- quantile(diffs, 0.75)
  iqr_d <- q3_d - q1_d
  mean_d <- mean(diffs)
  sd_d <- sd(diffs)
  
  if (all(diffs == 0)) {
    w_stat <- 0
    p_val <- 1.0
  } else {
    w_test <- tryCatch(wilcox.test(diffs, alternative = "two-sided", exact = FALSE), error = function(e) list(statistic = NA, p.value = NA))
    w_stat <- as.numeric(w_test$statistic)
    p_val <- as.numeric(w_test$p.value)
  }
  
  cat(sprintf("%-42s | N: %2d | Med Diff: %+.1f [IQR: %.1f] | Mean (SD): %+.2f (%.2f) | W = %4.1f, p = %.4f\n",
              col_label, n_pairs, med_d, iqr_d, mean_d, sd_d, w_stat, p_val))
  
  paired_results[[length(paired_results) + 1]] <- data.frame(
    Criterion = col_label,
    N_pairs = n_pairs,
    Median_Diff = med_d,
    Q1_Diff = q1_d,
    Q3_Diff = q3_d,
    IQR_Diff = iqr_d,
    Mean_Diff = round(mean_d, 2),
    SD_Diff = round(sd_d, 2),
    Wilcoxon_W = w_stat,
    p_value = round(p_val, 4),
    stringsAsFactors = FALSE
  )
}

df_paired <- bind_rows(paired_results)
write.csv(df_paired, file.path(opt$output_dir, "paired_differences_reproduced.csv"), row.names = FALSE)
cat("\nSaved Paired Differences reproduction to: paired_differences_reproduced.csv\n")

# 3. FORCED-CHOICE FORMAT PREFERENCE & ORDER BIAS ANALYSIS
if (file.exists(opt$preferences)) {
  cat("\n------------------------------------------------------------------------------\n")
  cat("PART 4: FORCED-CHOICE FORMAT PREFERENCE & PRESENTATION ORDER BIAS\n")
  cat("------------------------------------------------------------------------------\n")
  
  pref_raw <- read.csv(opt$preferences, stringsAsFactors = FALSE)
  # Extract Order and Preference columns across all evaluated blocks
  # In the comparison sheet, columns are grouped by evaluation sets
  pref_list <- list()
  
  for (start_col in seq(1, ncol(pref_raw), by = 5)) {
    if (start_col <= ncol(pref_raw)) {
      sub_cols <- pref_raw[, start_col:min(start_col + 1, ncol(pref_raw))]
      colnames(sub_cols) <- c("Order", "Preference")
      valid_rows <- sub_cols %>% filter(Order %in% c("Long-Short", "Short-Long") & Preference != "")
      pref_list[[length(pref_list) + 1]] <- valid_rows
    }
  }
  
  all_prefs <- bind_rows(pref_list)
  n_evals <- nrow(all_prefs)
  
  if (n_evals > 0) {
    all_prefs <- all_prefs %>%
      mutate(Preferred_Format = ifelse(grepl("Short", Preference), "Short-Form", "Long-Form"))
    
    t_pref <- table(all_prefs$Preferred_Format)
    cat(sprintf("Total Completed Format Evaluations: N = %d\n", n_evals))
    cat(sprintf("  - Preferred Short-Form (Psychologist Report): %d (%.1f%%)\n",
                t_pref["Short-Form"], 100 * t_pref["Short-Form"] / n_evals))
    cat(sprintf("  - Preferred Long-Form (Research Log):         %d (%.1f%%)\n",
                t_pref["Long-Form"], 100 * t_pref["Long-Form"] / n_evals))
    
    # 2x2 Contingency Table for Order vs Preference
    contingency <- table(all_prefs$Order, all_prefs$Preferred_Format)
    cat("\nContingency Table (Presentation Order vs Format Preference):\n")
    print(contingency)
    
    fisher_res <- fisher.test(contingency)
    cat(sprintf("\nFisher's Exact Test for Presentation Order Bias: p = %.3f\n", fisher_res$p.value))
    if (fisher_res$p.value > 0.05) {
      cat("Conclusion: No statistically significant presentation order bias detected.\n")
    } else {
      cat("Conclusion: Presentation order significantly influenced format choice.\n")
    }
  }
}

cat("\n==============================================================================\n")
cat("REPLICATION COMPLETED SUCCESSFULLY\n")
cat("==============================================================================\n")
