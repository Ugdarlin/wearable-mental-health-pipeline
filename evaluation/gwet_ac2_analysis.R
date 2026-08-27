# ==============================================================================
# Statistical Replication Script: Gwet's AC2 Inter-Rater Reliability Analysis
# Computes Gwet's AC2 coefficients with Linear Ordinal Weights and 95% CIs (N=7 raters)
# ==============================================================================

suppressPackageStartupMessages({
  if (!require("optparse", quietly = TRUE)) install.packages("optparse", repos="https://cloud.r-project.org")
  if (!require("tidyverse", quietly = TRUE)) install.packages("tidyverse", repos="https://cloud.r-project.org")
  library(optparse)
  library(tidyverse)
})

# Gwet's AC2 Calculation Function with Linear Ordinal Weights
calc_gwet_ac2 <- function(ratings_mat, q = 5) {
  # ratings_mat: N_items x n_raters matrix (values 1 to 5)
  N <- nrow(ratings_mat)
  n <- ncol(ratings_mat)
  
  # Linear weight matrix w_kl = 1 - |k - l| / (q - 1)
  w <- matrix(0, nrow = q, ncol = q)
  for (i in 1:q) {
    for (j in 1:q) {
      w[i, j] <- 1 - abs(i - j) / (q - 1)
    }
  }
  
  # Frequency matrix r_ik: count of raters assigning category k to item i
  r <- matrix(0, nrow = N, ncol = q)
  for (i in 1:N) {
    for (k in 1:q) {
      r[i, k] <- sum(ratings_mat[i, ] == k, na.rm = TRUE)
    }
  }
  
  n_i <- rowSums(r)
  valid <- n_i >= 2
  r <- r[valid, , drop = FALSE]
  n_i <- n_i[valid]
  N <- nrow(r)
  
  if (N == 0) return(list(ac2 = NA, se = NA, ci_lower = NA, ci_upper = NA))
  
  # Observed agreement P_a_i for each item
  P_a_i <- numeric(N)
  for (i in 1:N) {
    sum_w <- 0
    for (k in 1:q) {
      for (l in 1:q) {
        if (k == l) {
          sum_w <- sum_w + w[k, l] * r[i, k] * (r[i, k] - 1)
        } else {
          sum_w <- sum_w + w[k, l] * r[i, k] * r[i, l]
        }
      }
    }
    P_a_i[i] <- sum_w / (n_i[i] * (n_i[i] - 1))
  }
  P_a <- mean(P_a_i)
  
  # Chance agreement P_e
  pi_k <- colMeans(r / n_i)
  pi_k_star <- as.vector(w %*% pi_k)
  P_e <- sum(pi_k * (1 - pi_k_star)) / (q - 1)
  
  ac2 <- if (P_e == 1) 1 else (P_a - P_e) / (1 - P_e)
  
  # Variance & Standard Error (Gwet closed form)
  r_bar <- r / n_i
  r_bar_star <- r_bar %*% w
  P_e_i <- rowSums(r_bar * (1 - r_bar_star)) / (q - 1)
  num_i <- P_a_i - P_e - 2 * (1 - ac2) * (P_e_i - P_e)
  
  var_ac2 <- if (N > 1) (1 / (1 - P_e)^2) * sum((num_i - mean(num_i))^2) / (N * (N - 1)) else 0
  se_ac2 <- sqrt(max(0, var_ac2))
  
  ci_lower <- max(-1, ac2 - 1.96 * se_ac2)
  ci_upper <- min(1, ac2 + 1.96 * se_ac2)
  
  return(list(ac2 = ac2, se = se_ac2, ci_lower = ci_lower, ci_upper = ci_upper))
}

# Main Execution
option_list <- list(
  make_option(c("-i", "--input"), type="character", default="raw_expert_ratings.csv", help="Input CSV file")
)
opt <- parse_args(OptionParser(option_list=option_list))

cat("==============================================================================\n")
cat("GWET'S AC2 AGREEMENT ANALYSIS REPLICATION\n")
cat("Input File:", opt$input, "\n")
cat("==============================================================================\n\n")

data <- read.csv(opt$input, stringsAsFactors = FALSE)
metrics <- c(
  "A1_information_hierarchy" = "Information Hierarchy (A.1)",
  "A2_conciseness" = "Conciseness (A.2)",
  "B3_identification_of_factors" = "Identification of Factors (B.3)",
  "B4_likelihood_assessment" = "Likelihood Assessment (B.4)",
  "C5_relevance" = "Relevance (C.5)",
  "C6_appropriateness" = "Appropriateness (C.6)",
  "D7_overall_value_session_prep" = "Overall Value for Session Prep (D.7)"
)

results <- list()
for (form in c("Long_Form_Research_Log", "Short_Form_Psychologist_Report")) {
  cat(paste0("--- REPORT FORMAT: ", form, " ---\n"))
  sub_df <- data %>% filter(report_format == form)
  
  for (col_name in names(metrics)) {
    m_label <- metrics[[col_name]]
    
    # Construct N_items (3 weeks) x n_raters (7 raters) matrix
    mat_list <- list()
    for (wk in unique(sub_df$week)) {
      wk_sub <- sub_df %>% filter(week == wk)
      vals <- as.numeric(wk_sub[[col_name]])
      mat_list[[wk]] <- vals
    }
    ratings_matrix <- do.call(rbind, mat_list)
    
    all_vals <- as.numeric(sub_df[[col_name]])
    all_vals <- all_vals[!is.na(all_vals)]
    
    med <- median(all_vals)
    iqr <- IQR(all_vals)
    mean_val <- mean(all_vals)
    sd_val <- sd(all_vals)
    
    ac2_res <- calc_gwet_ac2(ratings_matrix, q = 5)
    
    cat(sprintf("%-38s | Median [IQR]: %.1f [%.1f] | Mean (SD): %.2f (%.2f) | AC2: %.4f (95%% CI: [%.3f, %.3f])\n",
                m_label, med, iqr, mean_val, sd_val, ac2_res$ac2, ac2_res$ci_lower, ac2_res$ci_upper))
  }
  cat("\n")
}
cat("Analysis finished successfully.\n")
