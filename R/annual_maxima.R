################################################################################
## ANNUAL MAXIMA CALCULATIONS
## Functions for computing annual maximum precipitation for various durations
################################################################################

source("R/utils.R")

#' Calculate moving window sums efficiently
#' @param x Numeric vector
#' @param window Window size
#' @return Vector of moving sums
#' @keywords internal
moving_window_sum <- function(x, window) {

  n <- length(x)

  if (window == 1) {
    return(x)
  }

  if (n < window) {
    return(numeric(0))
  }

  # Use efficient cumulative sum approach
  cumsum_x <- c(0, cumsum(x))

  # Calculate moving sums
  moving_sums <- cumsum_x[(window + 1):(n + 1)] - cumsum_x[1:(n - window + 1)]

  return(moving_sums)
}

#' Calculate annual maximum for single duration
#' @param year_data Vector of precipitation data for one year
#' @param duration_minutes Duration in minutes
#' @param interval_minutes Time resolution in minutes (default 5)
#' @return Maximum precipitation for the duration
#' @export
calculate_annual_max_single <- function(year_data, duration_minutes, interval_minutes = 5) {

  # Validate inputs
  if (!is.numeric(year_data) || length(year_data) == 0) {
    return(NA)
  }

  # Calculate window size
  window <- duration_minutes / interval_minutes

  if (window != round(window)) {
    warning(sprintf("Duration %d is not a multiple of interval %d",
                   duration_minutes, interval_minutes))
    window <- round(window)
  }

  if (window == 1) {
    # Direct maximum for single interval
    return(safe_max(year_data))
  }

  # Check sufficient data
  if (length(year_data) < window) {
    warning(sprintf("Insufficient data: %d records for window %d",
                   length(year_data), window))
    return(NA)
  }

  # Calculate moving window sums
  moving_sums <- moving_window_sum(year_data, window)

  return(safe_max(moving_sums))
}

#' Calculate annual maxima for all durations
#' @param df_long Data frame with columns: date, year, interval, precip
#' @param durations Vector of durations in minutes
#' @param duration_names Names for durations
#' @param verbose Print progress messages
#' @return Data frame with year and annual maxima for each duration
#' @export
calculate_annual_maxima <- function(df_long,
                                   durations = DURATIONS,
                                   duration_names = DURATION_NAMES,
                                   verbose = TRUE) {

  if (verbose) {
    print_header("CALCULATING ANNUAL MAXIMA")
  }

  # Get unique years
  years <- sort(unique(df_long$year))
  n_years <- length(years)
  n_durations <- length(durations)

  # Validate
  validate_years(years)

  # Pre-allocate result matrix
  annual_max <- matrix(NA, nrow = n_years, ncol = n_durations)
  colnames(annual_max) <- duration_names

  if (verbose) {
    cat(sprintf("Processing: %d years × %d durations\n", n_years, n_durations))
  }

  # Calculate for each year
  for (yr_idx in 1:n_years) {

    yr <- years[yr_idx]
    year_data <- df_long$precip[df_long$year == yr]

    if (verbose && (yr_idx == 1 || yr_idx %% 10 == 0 || yr_idx == n_years)) {
      cat(sprintf("\nYear %d (%d/%d):\n", yr, yr_idx, n_years))
      cat(sprintf("  Records: %d\n", length(year_data)))
      cat(sprintf("  Total precip: %.2f mm\n", sum(year_data, na.rm = TRUE)))
      cat(sprintf("  Max 5-min: %.2f mm\n", safe_max(year_data)))
    }

    # Calculate for each duration
    for (d_idx in 1:n_durations) {

      dur_min <- durations[d_idx]
      annual_max[yr_idx, d_idx] <- calculate_annual_max_single(
        year_data,
        dur_min,
        INTERVAL_MINUTES
      )

      # Verbose output for selected durations
      if (verbose && d_idx %in% c(1, 5, 9) && (yr_idx == 1 || yr_idx == n_years)) {
        cat(sprintf("    %6s: %6.2f mm\n",
                   duration_names[d_idx],
                   annual_max[yr_idx, d_idx]))
      }
    }
  }

  # Create result data frame
  result <- data.frame(year = years, annual_max, check.names = FALSE)

  if (verbose) {
    print_section("ANNUAL MAXIMA SUMMARY")
    cat("\nFirst 5 years:\n")
    print(head(result, 5))

    cat("\n\nLast 5 years:\n")
    print(tail(result, 5))

    # Overall statistics
    print_section("OVERALL STATISTICS (All Years)")
    for (d in 1:n_durations) {
      max_val <- safe_max(result[, d + 1])
      mean_val <- safe_mean(result[, d + 1])
      cat(sprintf("  %-6s: Max = %7.2f mm, Mean = %7.2f mm\n",
                 duration_names[d], max_val, mean_val))
    }

    # Monotonicity check
    print_section("MONOTONICITY CHECK")
    max_values <- sapply(2:ncol(result), function(i) safe_max(result[, i]))
    mono_check <- check_monotonic(max_values, duration_names)

    if (mono_check$is_monotonic) {
      cat("✓ Monotonicity PASSED: Longer durations have higher maxima\n")
    } else {
      cat("✗ Monotonicity FAILED: Detected violations:\n")
      for (v in mono_check$violations) {
        cat(sprintf("  %s (%.2f mm) > %s (%.2f mm)\n",
                   v$name1, v$value1, v$name2, v$value2))
      }
    }
  }

  return(result)
}

#' Get annual maxima for specific duration
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param duration_name Name of duration (e.g., "1hr")
#' @return Vector of annual maxima
#' @export
get_duration_maxima <- function(annual_max_df, duration_name) {

  if (!duration_name %in% names(annual_max_df)) {
    stop(sprintf("Duration '%s' not found in data. Available: %s",
                duration_name, paste(names(annual_max_df)[-1], collapse = ", ")))
  }

  return(annual_max_df[[duration_name]])
}

#' Export annual maxima to CSV
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param output_path Output CSV path
#' @param verbose Print confirmation message
#' @export
export_annual_maxima <- function(annual_max_df, output_path, verbose = TRUE) {

  write.csv(annual_max_df, output_path, row.names = FALSE)

  if (verbose) {
    cat(sprintf("\n✓ Annual maxima exported to: %s\n", output_path))
  }
}
