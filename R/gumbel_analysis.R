################################################################################
## GUMBEL DISTRIBUTION ANALYSIS
## Functions for fitting Gumbel distributions and calculating return levels
################################################################################

source("R/utils.R")

# Check if extRemes package is available
if (!requireNamespace("extRemes", quietly = TRUE)) {
  stop("Package 'extRemes' is required. Install with: install.packages('extRemes')")
}

library(extRemes)

#' Fit Gumbel distribution to annual maxima
#' @param data Vector of annual maximum values
#' @param duration_name Name of duration for reporting
#' @param method Fitting method ("MLE" or "GMLE")
#' @param verbose Print fit information
#' @return Fitted extRemes fevd object, or NULL if fit fails
#' @export
fit_gumbel <- function(data, duration_name = NULL, method = "MLE", verbose = TRUE) {

  # Remove NA values
  data_clean <- data[!is.na(data)]

  if (length(data_clean) < 10) {
    warning(sprintf("Insufficient data for %s: only %d values",
                   duration_name, length(data_clean)))
    return(NULL)
  }

  if (verbose && !is.null(duration_name)) {
    print_section(sprintf("Fitting Gumbel: %s", duration_name))
    cat(sprintf("  Data points: %d\n", length(data_clean)))
    cat(sprintf("  Mean: %.2f mm\n", mean(data_clean)))
    cat(sprintf("  SD: %.2f mm\n", sd(data_clean)))
    cat(sprintf("  Max: %.2f mm\n", max(data_clean)))
  }

  # Fit Gumbel distribution (GEV with shape = 0)
  fit <- tryCatch({
    fevd(data_clean, type = "Gumbel", method = method)
  }, error = function(e) {
    if (verbose) {
      cat(sprintf("  ERROR: %s\n", e$message))
    }
    return(NULL)
  })

  if (is.null(fit)) {
    return(NULL)
  }

  # Extract and display parameters
  if (verbose) {
    params <- findpars(fit)
    cat(sprintf("  Location (μ): %.3f\n", params$location))
    cat(sprintf("  Scale (σ): %.3f\n", params$scale))
    cat(sprintf("  Shape (ξ): %.3f (fixed for Gumbel)\n", 0.0))
  }

  return(fit)
}

#' Calculate return levels from fitted Gumbel distribution
#' @param fit Fitted fevd object from fit_gumbel
#' @param return_periods Vector of return periods (years)
#' @return Named vector of return levels
#' @export
calculate_return_levels <- function(fit, return_periods = RETURN_PERIODS) {

  if (is.null(fit)) {
    return(rep(NA, length(return_periods)))
  }

  tryCatch({
    rl <- return.level(fit, return.period = return_periods)
    names(rl) <- paste0("RL", return_periods)
    return(rl)
  }, error = function(e) {
    warning(sprintf("Error calculating return levels: %s", e$message))
    return(rep(NA, length(return_periods)))
  })
}

#' Fit Gumbel and calculate return levels for multiple durations
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param durations Vector of durations (uses DURATIONS if NULL)
#' @param duration_names Names for durations (uses DURATION_NAMES if NULL)
#' @param return_periods Return periods to calculate
#' @param method Fitting method
#' @param verbose Print progress
#' @return Data frame with fit results and return levels
#' @export
analyze_all_durations <- function(annual_max_df,
                                 durations = NULL,
                                 duration_names = NULL,
                                 return_periods = RETURN_PERIODS,
                                 method = "MLE",
                                 verbose = TRUE) {

  if (is.null(durations)) durations <- DURATIONS
  if (is.null(duration_names)) duration_names <- DURATION_NAMES

  if (verbose) {
    print_header("GUMBEL ANALYSIS - ALL DURATIONS")
  }

  n_durations <- length(duration_names)
  n_return_periods <- length(return_periods)

  # Initialize results data frame
  results <- data.frame(
    Duration = duration_names,
    Mean = NA,
    SD = NA,
    Max = NA,
    Location = NA,
    Scale = NA,
    Shape = 0.0  # Fixed for Gumbel
  )

  # Add columns for each return period
  for (rp in return_periods) {
    results[[paste0("RL", rp)]] <- NA
  }

  # Store fitted objects
  fitted_models <- list()

  # Fit each duration
  for (i in 1:n_durations) {

    dur_name <- duration_names[i]
    data_dur <- annual_max_df[[dur_name]]

    if (verbose) {
      cat("\n")
      cat(paste(rep("-", 60), collapse = ""), "\n")
      cat(sprintf("Duration %d/%d: %s\n", i, n_durations, dur_name))
      cat(paste(rep("-", 60), collapse = ""), "\n")
    }

    # Basic statistics
    results$Mean[i] <- safe_mean(data_dur)
    results$SD[i] <- sd(data_dur, na.rm = TRUE)
    results$Max[i] <- safe_max(data_dur)

    # Fit Gumbel
    fit <- fit_gumbel(data_dur, dur_name, method = method, verbose = verbose)
    fitted_models[[dur_name]] <- fit

    if (!is.null(fit)) {
      # Extract parameters
      params <- findpars(fit)
      results$Location[i] <- params$location
      results$Scale[i] <- params$scale

      # Calculate return levels
      rl <- calculate_return_levels(fit, return_periods)

      if (verbose) {
        cat("\n  Return Levels:\n")
        for (j in 1:length(return_periods)) {
          cat(sprintf("    %3d-year: %7.2f mm\n", return_periods[j], rl[j]))
        }
      }

      # Store in results
      for (j in 1:n_return_periods) {
        col_name <- paste0("RL", return_periods[j])
        results[[col_name]][i] <- rl[j]
      }
    }
  }

  if (verbose) {
    cat("\n")
    print_header("ANALYSIS COMPLETE")
    cat("\nParameter Summary:\n")
    print(results[, c("Duration", "Mean", "SD", "Location", "Scale")])

    cat("\n\nReturn Level Summary:\n")
    rl_cols <- grep("^RL", names(results), value = TRUE)
    print(results[, c("Duration", rl_cols)])
  }

  # Return both results and fitted models
  attr(results, "fitted_models") <- fitted_models

  return(results)
}

#' Perform quality checks on return level results
#' @param results Data frame from analyze_all_durations
#' @param return_period Return period to check (default 100)
#' @param verbose Print check results
#' @return List with check results
#' @export
check_return_levels <- function(results, return_period = 100, verbose = TRUE) {

  if (verbose) {
    print_header("QUALITY CHECKS")
  }

  rl_col <- paste0("RL", return_period)

  if (!rl_col %in% names(results)) {
    stop(sprintf("Return period %d not found in results", return_period))
  }

  rl_values <- results[[rl_col]]

  # Check 1: Monotonicity
  mono_check <- check_monotonic(rl_values, results$Duration)

  if (verbose) {
    print_section(sprintf("Check 1: Monotonicity (RL%d)", return_period))
    if (mono_check$is_monotonic) {
      cat("✓ PASSED: Return levels increase with duration\n")
    } else {
      cat("✗ FAILED: Detected violations:\n")
      for (v in mono_check$violations) {
        cat(sprintf("  %s (%.2f mm) > %s (%.2f mm)\n",
                   v$name1, v$value1, v$name2, v$value2))
      }
    }
  }

  # Check 2: Return level vs observed maximum
  if (verbose) {
    print_section("Check 2: Return Level vs Observed Max")
  }

  rl_vs_max_check <- list()
  for (i in 1:nrow(results)) {
    rl <- results[[rl_col]][i]
    obs_max <- results$Max[i]

    if (!is.na(rl) && !is.na(obs_max)) {
      ratio <- rl / obs_max
      status <- if (ratio >= 1.0) "OK" else "WARNING"

      rl_vs_max_check[[i]] <- list(
        duration = results$Duration[i],
        rl = rl,
        obs_max = obs_max,
        ratio = ratio,
        status = status
      )

      if (verbose) {
        cat(sprintf("  %-6s: RL%d=%.2f, Max=%.2f, Ratio=%.2f [%s]\n",
                   results$Duration[i], return_period, rl, obs_max, ratio, status))
      }
    }
  }

  # Check 3: Scale parameter reasonableness
  if (verbose) {
    print_section("Check 3: Scale Parameters")
  }

  scale_check <- list()
  for (i in 1:nrow(results)) {
    scale <- results$Scale[i]
    sd_obs <- results$SD[i]

    if (!is.na(scale) && !is.na(sd_obs)) {
      # For Gumbel, σ ≈ SD / 1.282
      expected_scale <- sd_obs / 1.282
      ratio <- scale / expected_scale

      scale_check[[i]] <- list(
        duration = results$Duration[i],
        scale = scale,
        sd = sd_obs,
        expected_scale = expected_scale,
        ratio = ratio
      )

      if (verbose) {
        cat(sprintf("  %-6s: σ=%.2f, SD=%.2f, Expected σ=%.2f, Ratio=%.2f\n",
                   results$Duration[i], scale, sd_obs, expected_scale, ratio))
      }
    }
  }

  return(list(
    monotonicity = mono_check,
    rl_vs_max = rl_vs_max_check,
    scale_check = scale_check
  ))
}

#' Export Gumbel analysis results
#' @param results Data frame from analyze_all_durations
#' @param output_path Output CSV path
#' @param verbose Print confirmation
#' @export
export_gumbel_results <- function(results, output_path, verbose = TRUE) {

  # Remove fitted models attribute before export
  results_clean <- results
  attr(results_clean, "fitted_models") <- NULL

  write.csv(results_clean, output_path, row.names = FALSE)

  if (verbose) {
    cat(sprintf("\n✓ Gumbel analysis results exported to: %s\n", output_path))
  }
}
