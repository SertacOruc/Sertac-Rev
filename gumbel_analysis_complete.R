################################################################################
## COMPLETE GUMBEL EXTREME RAINFALL ANALYSIS
## Single run-ready script with all functions included
##
## FEATURES:
## - Stationary Gumbel analysis with constant parameters
## - Nonstationary Gumbel analysis with time-varying parameters (trend analysis)
## - Model comparison (AIC-based)
## - Comprehensive visualizations (IDF curves, return levels, trends)
##
## INSTRUCTIONS:
## 1. Install required package: install.packages("extRemes")
## 2. Edit the CONFIGURATION section below (lines 20-50)
## 3. Set RUN_NONSTATIONARY = TRUE to enable nonstationary analysis
## 4. Run the entire script
## 5. Results will be saved to the output/ folder
################################################################################

library(extRemes)

################################################################################
## CONFIGURATION - EDIT THIS SECTION
################################################################################

# Data parameters
DATA_DIR <- "C:/Users/ser_o/OneDrive/Documents/EXTREMES/hyetos/hyetos"
START_YEAR <- 2015
END_YEAR <- 2100
SCENARIO <- "ssp245"
STATION_ID <- 31

# Output settings
OUTPUT_DIR <- "output"
EXPORT_ANNUAL_MAX <- TRUE
EXPORT_RESULTS <- TRUE
EXPORT_PLOTS <- TRUE
VERBOSE <- TRUE

# Analysis parameters (modify if needed)
DURATIONS <- c(5, 10, 15, 30, 60, 180, 360, 720, 1440)
DURATION_NAMES <- c("5min", "10min", "15min", "30min", "1hr", "3hr", "6hr", "12hr", "24hr")
RETURN_PERIODS <- c(2, 5, 10, 25, 50, 100)
INTERVALS_PER_DAY <- 288
DAYS_PER_MONTH <- 30
MONTHS_PER_YEAR <- 12
INTERVAL_MINUTES <- 5

# Nonstationary analysis parameters
RUN_NONSTATIONARY <- TRUE  # Set to TRUE to perform nonstationary analysis
NONSTAT_TREND_LOCATION <- TRUE  # Allow linear trend in location parameter
NONSTAT_TREND_SCALE <- FALSE     # Allow linear trend in scale parameter (set FALSE for stability)

################################################################################
## UTILITY FUNCTIONS
################################################################################

print_header <- function(title) {
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat(paste0("  ", title, "\n"))
  cat(paste(rep("=", 80), collapse = ""), "\n")
}

print_section <- function(title) {
  cat(sprintf("\n--- %s ---\n", title))
}

safe_max <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA)
  return(max(x, na.rm = TRUE))
}

safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) return(NA)
  return(mean(x, na.rm = TRUE))
}

validate_precip_data <- function(data) {
  if (!is.numeric(data)) stop("Precipitation data must be numeric")
  if (any(data < 0, na.rm = TRUE)) {
    warning("Negative precipitation values detected. Setting to 0.")
    data[data < 0] <- 0
  }
  return(TRUE)
}

validate_years <- function(years, min_years = 10) {
  if (!is.numeric(years) || length(years) < min_years) {
    stop(sprintf("Insufficient years: %d. Need at least %d years",
                 length(years), min_years))
  }
  return(TRUE)
}

create_360day_calendar <- function(start_year, end_year, intervals_per_day = 288) {
  years <- start_year:end_year
  calendar_vec <- c()

  for (yy in years) {
    days_360 <- as.Date(paste0(yy, "-01-01")) + 0:359
    days_360_rep <- rep(days_360, each = intervals_per_day)
    calendar_vec <- c(calendar_vec, days_360_rep)
  }

  return(calendar_vec)
}

check_monotonic <- function(values, names = NULL, tolerance = 1e-6) {
  if (is.null(names)) names <- paste0("Val", seq_along(values))

  violations <- list()
  is_monotonic <- TRUE

  for (i in 1:(length(values) - 1)) {
    if (!is.na(values[i]) && !is.na(values[i + 1])) {
      if (values[i] > values[i + 1] * (1 + tolerance)) {
        is_monotonic <- FALSE
        violations[[length(violations) + 1]] <- list(
          position = i,
          name1 = names[i],
          value1 = values[i],
          name2 = names[i + 1],
          value2 = values[i + 1]
        )
      }
    }
  }

  return(list(is_monotonic = is_monotonic, violations = violations))
}

################################################################################
## DATA LOADING FUNCTIONS
################################################################################

load_hyetos_month <- function(data_dir, month, scenario = "ssp245",
                               station_id = 31, n_intervals = 288) {

  filename <- sprintf("hyetosout_%s_ %d _ %d _ %d .txt",
                     scenario, station_id, month, station_id)
  filepath <- file.path(data_dir, filename)

  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }

  tryCatch({
    data_matrix <- as.matrix(read.table(filepath))[, 5:(4 + n_intervals)]
    return(data_matrix)
  }, error = function(e) {
    stop(sprintf("Error reading file %s: %s", filepath, e$message))
  })
}

load_precipitation_data <- function(data_dir, start_year, end_year,
                                   scenario = "ssp245", station_id = 31,
                                   verbose = TRUE) {

  if (verbose) print_header("LOADING PRECIPITATION DATA")

  n_years <- end_year - start_year + 1
  n_months <- MONTHS_PER_YEAR
  total_records <- n_years * DAYS_PER_MONTH * n_months * INTERVALS_PER_DAY

  yearly_data <- data.frame(
    date = as.Date(character(total_records)),
    year = integer(total_records),
    interval = integer(total_records),
    precip = numeric(total_records)
  )

  record_idx <- 1

  for (year_idx in 1:n_years) {
    year <- start_year + year_idx - 1

    for (month in 1:n_months) {

      if (verbose && month == 1) {
        cat(sprintf("Loading year %d (%d/%d)...\n", year, year_idx, n_years))
      }

      month_data <- load_hyetos_month(data_dir, month, scenario, station_id)

      n_days_in_file <- nrow(month_data)
      data_filled <- matrix(0, nrow = DAYS_PER_MONTH * n_years, ncol = INTERVALS_PER_DAY)

      if (n_days_in_file > 0) {
        data_filled[1:n_days_in_file, ] <- month_data
      }

      year_start_row <- (year_idx - 1) * DAYS_PER_MONTH + 1
      year_end_row <- year_idx * DAYS_PER_MONTH
      year_month_data <- data_filled[year_start_row:year_end_row, ]

      for (day in 1:DAYS_PER_MONTH) {
        day_data <- year_month_data[day, ]

        idx_start <- record_idx
        idx_end <- record_idx + INTERVALS_PER_DAY - 1

        yearly_data$precip[idx_start:idx_end] <- day_data
        yearly_data$year[idx_start:idx_end] <- year
        yearly_data$interval[idx_start:idx_end] <- 1:INTERVALS_PER_DAY

        record_idx <- idx_end + 1
      }
    }
  }

  calendar_vec <- create_360day_calendar(start_year, end_year, INTERVALS_PER_DAY)
  yearly_data$date <- calendar_vec
  yearly_data$precip[is.na(yearly_data$precip)] <- 0

  validate_precip_data(yearly_data$precip)

  if (verbose) {
    cat(sprintf("\nData loaded successfully:\n"))
    cat(sprintf("  Years: %d - %d (%d years)\n", start_year, end_year, n_years))
    cat(sprintf("  Total records: %d\n", nrow(yearly_data)))
    cat(sprintf("  Total precipitation: %.2f mm\n", sum(yearly_data$precip)))
    cat(sprintf("  Mean precipitation: %.4f mm per 5-min\n", mean(yearly_data$precip)))
    cat(sprintf("  Max precipitation: %.2f mm per 5-min\n", max(yearly_data$precip)))
  }

  return(yearly_data)
}

################################################################################
## ANNUAL MAXIMA FUNCTIONS
################################################################################

moving_window_sum <- function(x, window) {
  n <- length(x)

  if (window == 1) return(x)
  if (n < window) return(numeric(0))

  cumsum_x <- c(0, cumsum(x))
  moving_sums <- cumsum_x[(window + 1):(n + 1)] - cumsum_x[1:(n - window + 1)]

  return(moving_sums)
}

calculate_annual_max_single <- function(year_data, duration_minutes,
                                       interval_minutes = 5) {

  if (!is.numeric(year_data) || length(year_data) == 0) return(NA)

  window <- duration_minutes / interval_minutes

  if (window != round(window)) {
    warning(sprintf("Duration %d is not a multiple of interval %d",
                   duration_minutes, interval_minutes))
    window <- round(window)
  }

  if (window == 1) return(safe_max(year_data))

  if (length(year_data) < window) {
    warning(sprintf("Insufficient data: %d records for window %d",
                   length(year_data), window))
    return(NA)
  }

  moving_sums <- moving_window_sum(year_data, window)
  return(safe_max(moving_sums))
}

calculate_annual_maxima <- function(df_long, durations = DURATIONS,
                                   duration_names = DURATION_NAMES,
                                   verbose = TRUE) {

  if (verbose) print_header("CALCULATING ANNUAL MAXIMA")

  years <- sort(unique(df_long$year))
  n_years <- length(years)
  n_durations <- length(durations)

  validate_years(years)

  annual_max <- matrix(NA, nrow = n_years, ncol = n_durations)
  colnames(annual_max) <- duration_names

  if (verbose) {
    cat(sprintf("Processing: %d years × %d durations\n", n_years, n_durations))
  }

  for (yr_idx in 1:n_years) {
    yr <- years[yr_idx]
    year_data <- df_long$precip[df_long$year == yr]

    if (verbose && (yr_idx == 1 || yr_idx %% 10 == 0 || yr_idx == n_years)) {
      cat(sprintf("\nYear %d (%d/%d):\n", yr, yr_idx, n_years))
      cat(sprintf("  Records: %d\n", length(year_data)))
      cat(sprintf("  Total precip: %.2f mm\n", sum(year_data, na.rm = TRUE)))
      cat(sprintf("  Max 5-min: %.2f mm\n", safe_max(year_data)))
    }

    for (d_idx in 1:n_durations) {
      dur_min <- durations[d_idx]
      annual_max[yr_idx, d_idx] <- calculate_annual_max_single(
        year_data, dur_min, INTERVAL_MINUTES
      )

      if (verbose && d_idx %in% c(1, 5, 9) && (yr_idx == 1 || yr_idx == n_years)) {
        cat(sprintf("    %6s: %6.2f mm\n",
                   duration_names[d_idx], annual_max[yr_idx, d_idx]))
      }
    }
  }

  result <- data.frame(year = years, annual_max, check.names = FALSE)

  if (verbose) {
    print_section("ANNUAL MAXIMA SUMMARY")
    cat("\nFirst 5 years:\n")
    print(head(result, 5))

    cat("\n\nLast 5 years:\n")
    print(tail(result, 5))

    print_section("OVERALL STATISTICS (All Years)")
    for (d in 1:n_durations) {
      max_val <- safe_max(result[, d + 1])
      mean_val <- safe_mean(result[, d + 1])
      cat(sprintf("  %-6s: Max = %7.2f mm, Mean = %7.2f mm\n",
                 duration_names[d], max_val, mean_val))
    }

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

################################################################################
## GUMBEL ANALYSIS FUNCTIONS
################################################################################

fit_gumbel <- function(data, duration_name = NULL, method = "MLE",
                      verbose = TRUE) {

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

  fit <- tryCatch({
    fevd(data_clean, type = "Gumbel", method = method)
  }, error = function(e) {
    if (verbose) cat(sprintf("  ERROR: %s\n", e$message))
    return(NULL)
  })

  if (is.null(fit)) return(NULL)

  if (verbose) {
    params <- findpars(fit)
    cat(sprintf("  Location (μ): %.3f\n", params$location))
    cat(sprintf("  Scale (σ): %.3f\n", params$scale))
    cat(sprintf("  Shape (ξ): %.3f (fixed for Gumbel)\n", 0.0))
  }

  return(fit)
}

calculate_return_levels <- function(fit, return_periods = RETURN_PERIODS) {
  if (is.null(fit)) return(rep(NA, length(return_periods)))

  tryCatch({
    rl <- return.level(fit, return.period = return_periods)
    names(rl) <- paste0("RL", return_periods)
    return(rl)
  }, error = function(e) {
    warning(sprintf("Error calculating return levels: %s", e$message))
    return(rep(NA, length(return_periods)))
  })
}

analyze_all_durations <- function(annual_max_df, durations = DURATIONS,
                                 duration_names = DURATION_NAMES,
                                 return_periods = RETURN_PERIODS,
                                 method = "MLE", verbose = TRUE) {

  if (verbose) print_header("GUMBEL ANALYSIS - ALL DURATIONS")

  n_durations <- length(duration_names)
  n_return_periods <- length(return_periods)

  results <- data.frame(
    Duration = duration_names,
    Mean = NA,
    SD = NA,
    Max = NA,
    Location = NA,
    Scale = NA,
    Shape = 0.0
  )

  for (rp in return_periods) {
    results[[paste0("RL", rp)]] <- NA
  }

  fitted_models <- list()

  for (i in 1:n_durations) {
    dur_name <- duration_names[i]
    data_dur <- annual_max_df[[dur_name]]

    if (verbose) {
      cat("\n")
      cat(paste(rep("-", 60), collapse = ""), "\n")
      cat(sprintf("Duration %d/%d: %s\n", i, n_durations, dur_name))
      cat(paste(rep("-", 60), collapse = ""), "\n")
    }

    results$Mean[i] <- safe_mean(data_dur)
    results$SD[i] <- sd(data_dur, na.rm = TRUE)
    results$Max[i] <- safe_max(data_dur)

    fit <- fit_gumbel(data_dur, dur_name, method = method, verbose = verbose)
    fitted_models[[dur_name]] <- fit

    if (!is.null(fit)) {
      params <- findpars(fit)
      results$Location[i] <- params$location
      results$Scale[i] <- params$scale

      rl <- calculate_return_levels(fit, return_periods)

      if (verbose) {
        cat("\n  Return Levels:\n")
        for (j in 1:length(return_periods)) {
          cat(sprintf("    %3d-year: %7.2f mm\n", return_periods[j], rl[j]))
        }
      }

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

  attr(results, "fitted_models") <- fitted_models
  return(results)
}

check_return_levels <- function(results, return_period = 100, verbose = TRUE) {
  if (verbose) print_header("QUALITY CHECKS")

  rl_col <- paste0("RL", return_period)

  if (!rl_col %in% names(results)) {
    stop(sprintf("Return period %d not found in results", return_period))
  }

  rl_values <- results[[rl_col]]
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

    print_section("Check 2: Return Level vs Observed Max")
    for (i in 1:nrow(results)) {
      rl <- results[[rl_col]][i]
      obs_max <- results$Max[i]

      if (!is.na(rl) && !is.na(obs_max)) {
        ratio <- rl / obs_max
        status <- if (ratio >= 1.0) "OK" else "WARNING"
        cat(sprintf("  %-6s: RL%d=%.2f, Max=%.2f, Ratio=%.2f [%s]\n",
                   results$Duration[i], return_period, rl, obs_max, ratio, status))
      }
    }
  }

  return(list(monotonicity = mono_check))
}

################################################################################
## NONSTATIONARY ANALYSIS FUNCTIONS
################################################################################

fit_gumbel_nonstationary <- function(data, years, duration_name = NULL,
                                     trend_location = TRUE, trend_scale = FALSE,
                                     method = "MLE", verbose = TRUE) {

  data_clean <- data[!is.na(data)]
  years_clean <- years[!is.na(data)]

  if (length(data_clean) < 15) {
    warning(sprintf("Insufficient data for nonstationary %s: only %d values",
                   duration_name, length(data_clean)))
    return(NULL)
  }

  # Normalize years to start from 0 for numerical stability
  years_normalized <- years_clean - min(years_clean)

  if (verbose && !is.null(duration_name)) {
    print_section(sprintf("Fitting Nonstationary Gumbel: %s", duration_name))
    cat(sprintf("  Data points: %d\n", length(data_clean)))
    cat(sprintf("  Year range: %d - %d\n", min(years_clean), max(years_clean)))
    cat(sprintf("  Trend in location: %s\n", ifelse(trend_location, "YES", "NO")))
    cat(sprintf("  Trend in scale: %s\n", ifelse(trend_scale, "YES", "NO")))
  }

  # Build formulas for location and scale
  location_formula <- if (trend_location) {
    ~ years_normalized
  } else {
    ~ 1
  }

  scale_formula <- if (trend_scale) {
    ~ years_normalized
  } else {
    ~ 1
  }

  fit <- tryCatch({
    fevd(data_clean,
         type = "Gumbel",
         method = method,
         location.fun = location_formula,
         scale.fun = scale_formula)
  }, error = function(e) {
    if (verbose) cat(sprintf("  ERROR: %s\n", e$message))
    return(NULL)
  })

  if (is.null(fit)) return(NULL)

  if (verbose) {
    params <- fit$results$par
    cat(sprintf("  Location (μ0): %.3f\n", params[1]))

    if (trend_location && length(params) >= 2) {
      cat(sprintf("  Location trend (μ1): %.6f per year\n", params[2]))
      total_change <- params[2] * (max(years_clean) - min(years_clean))
      cat(sprintf("  Total location change: %.3f mm over %d years\n",
                 total_change, max(years_clean) - min(years_clean)))
    }

    scale_idx <- if (trend_location) 3 else 2
    cat(sprintf("  Scale (σ0): %.3f\n", params[scale_idx]))

    if (trend_scale && length(params) >= scale_idx + 1) {
      cat(sprintf("  Scale trend (σ1): %.6f per year\n", params[scale_idx + 1]))
    }
  }

  # Store year information for later use
  attr(fit, "years") <- years_clean
  attr(fit, "years_normalized") <- years_normalized
  attr(fit, "year_min") <- min(years_clean)
  attr(fit, "trend_location") <- trend_location
  attr(fit, "trend_scale") <- trend_scale

  return(fit)
}

calculate_return_levels_nonstationary <- function(fit, return_periods = RETURN_PERIODS,
                                                  target_years = NULL) {

  if (is.null(fit)) return(NULL)

  years <- attr(fit, "years")
  year_min <- attr(fit, "year_min")
  trend_location <- attr(fit, "trend_location")
  trend_scale <- attr(fit, "trend_scale")

  # If no target years specified, use first, middle, and last year
  if (is.null(target_years)) {
    target_years <- c(min(years), median(years), max(years))
  }

  results <- list()

  for (target_year in target_years) {
    years_normalized <- target_year - year_min

    tryCatch({
      rl <- return.level(fit,
                        return.period = return_periods,
                        do.ci = FALSE)

      # For nonstationary models, we need to evaluate at specific time point
      if (trend_location || trend_scale) {
        # Get parameters
        params <- fit$results$par

        # Calculate location at target year
        mu <- params[1]
        if (trend_location && length(params) >= 2) {
          mu <- mu + params[2] * years_normalized
        }

        # Calculate scale at target year
        scale_idx <- if (trend_location) 3 else 2
        sigma <- params[scale_idx]
        if (trend_scale && length(params) >= scale_idx + 1) {
          sigma <- sigma + params[scale_idx + 1] * years_normalized
        }

        # Calculate return levels using Gumbel quantile function
        # RL = μ - σ * log(-log(1 - 1/T))
        for (i in seq_along(return_periods)) {
          p <- 1 - 1/return_periods[i]
          rl[i] <- mu - sigma * log(-log(p))
        }
      }

      names(rl) <- paste0("RL", return_periods)
      results[[as.character(target_year)]] <- rl
    }, error = function(e) {
      warning(sprintf("Error calculating return levels for year %d: %s",
                     target_year, e$message))
      results[[as.character(target_year)]] <- rep(NA, length(return_periods))
    })
  }

  return(results)
}

analyze_all_durations_nonstationary <- function(annual_max_df,
                                               durations = DURATIONS,
                                               duration_names = DURATION_NAMES,
                                               return_periods = RETURN_PERIODS,
                                               trend_location = TRUE,
                                               trend_scale = FALSE,
                                               method = "MLE",
                                               verbose = TRUE) {

  if (verbose) print_header("NONSTATIONARY GUMBEL ANALYSIS - ALL DURATIONS")

  n_durations <- length(duration_names)
  years <- annual_max_df$year

  # Target years for return level calculation
  target_years <- c(min(years), round(median(years)), max(years))

  # Initialize results data frame
  results <- data.frame(
    Duration = duration_names,
    Location_mu0 = NA,
    Location_mu1 = NA,
    Scale_sigma0 = NA,
    Scale_sigma1 = NA,
    AIC = NA,
    Total_Change = NA
  )

  # Add columns for return levels at different time points
  for (year in target_years) {
    for (rp in return_periods) {
      col_name <- sprintf("RL%d_Year%d", rp, year)
      results[[col_name]] <- NA
    }
  }

  fitted_models <- list()

  for (i in 1:n_durations) {
    dur_name <- duration_names[i]
    data_dur <- annual_max_df[[dur_name]]

    if (verbose) {
      cat("\n")
      cat(paste(rep("-", 60), collapse = ""), "\n")
      cat(sprintf("Duration %d/%d: %s\n", i, n_durations, dur_name))
      cat(paste(rep("-", 60), collapse = ""), "\n")
    }

    fit <- fit_gumbel_nonstationary(data_dur, years, dur_name,
                                    trend_location = trend_location,
                                    trend_scale = trend_scale,
                                    method = method,
                                    verbose = verbose)

    fitted_models[[dur_name]] <- fit

    if (!is.null(fit)) {
      params <- fit$results$par

      # Extract parameters
      results$Location_mu0[i] <- params[1]
      if (trend_location && length(params) >= 2) {
        results$Location_mu1[i] <- params[2]
        year_range <- max(years) - min(years)
        results$Total_Change[i] <- params[2] * year_range
      }

      scale_idx <- if (trend_location) 3 else 2
      results$Scale_sigma0[i] <- params[scale_idx]

      if (trend_scale && length(params) >= scale_idx + 1) {
        results$Scale_sigma1[i] <- params[scale_idx + 1]
      }

      # Calculate AIC if available
      if (!is.null(fit$results$value)) {
        n_params <- length(params)
        log_lik <- -fit$results$value
        results$AIC[i] <- 2 * n_params - 2 * log_lik
      }

      # Calculate return levels for different time points
      rl_list <- calculate_return_levels_nonstationary(fit, return_periods, target_years)

      if (verbose) {
        cat("\n  Return Levels by Time Period:\n")
      }

      for (year in target_years) {
        rl <- rl_list[[as.character(year)]]

        if (verbose) {
          cat(sprintf("\n  Year %d:\n", year))
          for (j in 1:length(return_periods)) {
            cat(sprintf("    %3d-year: %7.2f mm\n", return_periods[j], rl[j]))
          }
        }

        for (j in 1:length(return_periods)) {
          col_name <- sprintf("RL%d_Year%d", return_periods[j], year)
          results[[col_name]][i] <- rl[j]
        }
      }
    }
  }

  if (verbose) {
    cat("\n")
    print_header("NONSTATIONARY ANALYSIS COMPLETE")
    cat("\nTrend Summary:\n")
    trend_cols <- c("Duration", "Location_mu0", "Location_mu1", "Total_Change", "Scale_sigma0")
    print(results[, intersect(trend_cols, names(results))])
  }

  attr(results, "fitted_models") <- fitted_models
  attr(results, "target_years") <- target_years
  attr(results, "trend_location") <- trend_location
  attr(results, "trend_scale") <- trend_scale

  return(results)
}

compare_stationary_nonstationary <- function(stat_results, nonstat_results,
                                            annual_max_df, verbose = TRUE) {

  if (verbose) print_header("STATIONARY vs NONSTATIONARY COMPARISON")

  stat_models <- attr(stat_results, "fitted_models")
  nonstat_models <- attr(nonstat_results, "fitted_models")

  comparison <- data.frame(
    Duration = stat_results$Duration,
    Stat_AIC = NA,
    Nonstat_AIC = NA,
    Delta_AIC = NA,
    Preferred = NA
  )

  for (i in 1:nrow(comparison)) {
    dur_name <- comparison$Duration[i]

    # Calculate AIC for stationary model
    stat_fit <- stat_models[[dur_name]]
    if (!is.null(stat_fit) && !is.null(stat_fit$results$value)) {
      n_params_stat <- 2  # location and scale
      log_lik_stat <- -stat_fit$results$value
      comparison$Stat_AIC[i] <- 2 * n_params_stat - 2 * log_lik_stat
    }

    # Get AIC for nonstationary model
    comparison$Nonstat_AIC[i] <- nonstat_results$AIC[i]

    # Calculate delta AIC
    if (!is.na(comparison$Stat_AIC[i]) && !is.na(comparison$Nonstat_AIC[i])) {
      comparison$Delta_AIC[i] <- comparison$Nonstat_AIC[i] - comparison$Stat_AIC[i]

      # Preferred model (delta AIC < -2 suggests nonstationary is better)
      if (comparison$Delta_AIC[i] < -2) {
        comparison$Preferred[i] <- "Nonstationary"
      } else if (comparison$Delta_AIC[i] > 2) {
        comparison$Preferred[i] <- "Stationary"
      } else {
        comparison$Preferred[i] <- "Similar"
      }
    }
  }

  if (verbose) {
    cat("\nAIC Comparison (lower is better):\n")
    cat("Delta AIC < -2: Nonstationary preferred\n")
    cat("Delta AIC > +2: Stationary preferred\n\n")
    print(comparison)

    # Summary
    cat("\n\nSummary:\n")
    cat(sprintf("  Nonstationary preferred: %d durations\n",
               sum(comparison$Preferred == "Nonstationary", na.rm = TRUE)))
    cat(sprintf("  Stationary preferred: %d durations\n",
               sum(comparison$Preferred == "Stationary", na.rm = TRUE)))
    cat(sprintf("  Similar performance: %d durations\n",
               sum(comparison$Preferred == "Similar", na.rm = TRUE)))
  }

  return(comparison)
}

################################################################################
## VISUALIZATION FUNCTIONS
################################################################################

plot_return_levels <- function(results, return_period = 100,
                              col = "steelblue", main = NULL) {

  rl_col <- paste0("RL", return_period)

  if (!rl_col %in% names(results)) {
    stop(sprintf("Return period %d not found in results", return_period))
  }

  if (is.null(main)) {
    main <- sprintf("%d-Year Return Levels (Gumbel)", return_period)
  }

  barplot(results[[rl_col]],
         names.arg = results$Duration,
         col = col,
         main = main,
         ylab = "Precipitation (mm)",
         las = 2,
         border = NA)

  grid(nx = NA, ny = NULL, col = "gray90", lty = 1)
}

plot_scale_parameters <- function(results, col = "lightblue") {
  barplot(results$Scale,
         names.arg = results$Duration,
         col = col,
         main = "Scale Parameters (σ)",
         ylab = "Scale (σ)",
         las = 2,
         border = NA)

  grid(nx = NA, ny = NULL, col = "gray90", lty = 1)
}

plot_mean_vs_return_level <- function(results, return_period = 100) {
  rl_col <- paste0("RL", return_period)

  plot(results$Mean, results[[rl_col]],
      pch = 19, col = "darkblue", cex = 1.5,
      xlab = "Mean Annual Maximum (mm)",
      ylab = sprintf("RL%d (mm)", return_period),
      main = sprintf("Mean vs RL%d", return_period))

  text(results$Mean, results[[rl_col]],
      labels = results$Duration,
      pos = 3, cex = 0.8, col = "darkblue")

  abline(0, 1, col = "red", lty = 2, lwd = 2)

  legend("topleft",
        legend = c("Data", "1:1 Line"),
        col = c("darkblue", "red"),
        pch = c(19, NA),
        lty = c(NA, 2),
        lwd = c(NA, 2),
        bty = "n")

  grid(col = "gray90", lty = 1)
}

plot_idf_curve <- function(results, return_periods = c(2, 10, 25, 50, 100),
                          colors = NULL, log_scale = TRUE) {

  if (is.null(colors)) {
    colors <- c("darkgreen", "blue", "orange", "red", "darkred")
  }

  durations <- DURATIONS[match(results$Duration, DURATION_NAMES)]

  log_axis <- if (log_scale) "xy" else ""

  plot(1, type = "n",
      xlim = range(durations),
      ylim = range(results[, grep("^RL", names(results))], na.rm = TRUE),
      log = log_axis,
      xlab = "Duration (minutes)",
      ylab = "Precipitation (mm)",
      main = "IDF Curves - Gumbel Analysis",
      las = 1)

  for (i in seq_along(return_periods)) {
    rp <- return_periods[i]
    rl_col <- paste0("RL", rp)

    if (rl_col %in% names(results)) {
      lines(durations, results[[rl_col]],
           type = "b", pch = 19, col = colors[i], lwd = 2)
    }
  }

  legend("topleft",
        legend = paste0(return_periods, "-year"),
        col = colors[1:length(return_periods)],
        lwd = 2,
        pch = 19,
        bty = "n",
        title = "Return Period")

  grid(col = "gray80", lty = 1)
}

plot_summary <- function(results, annual_max_df = NULL, return_period = 100) {
  par(mfrow = c(2, 2), mar = c(5, 4, 3, 2))

  plot_return_levels(results, return_period = return_period)
  plot_scale_parameters(results)
  plot_mean_vs_return_level(results, return_period = return_period)
  plot_idf_curve(results, log_scale = TRUE)

  par(mfrow = c(1, 1))
}

export_plots_pdf <- function(results, annual_max_df = NULL,
                            output_path = "gumbel_analysis_plots.pdf",
                            return_period = 100, verbose = TRUE) {

  pdf(output_path, width = 11, height = 8.5)

  plot_summary(results, annual_max_df, return_period)

  par(mfrow = c(1, 1), mar = c(5, 5, 4, 2))
  plot_idf_curve(results, return_periods = c(2, 5, 10, 25, 50, 100))

  dev.off()

  if (verbose) {
    cat(sprintf("\n✓ Plots exported to: %s\n", output_path))
  }
}

plot_nonstationary_trends <- function(nonstat_results, main = "Location Parameter Trends") {

  has_trend <- !is.na(nonstat_results$Location_mu1)

  if (!any(has_trend)) {
    cat("No trend data available to plot\n")
    return(invisible(NULL))
  }

  # Plot total change
  barplot(nonstat_results$Total_Change,
         names.arg = nonstat_results$Duration,
         col = ifelse(nonstat_results$Total_Change > 0, "darkred", "steelblue"),
         main = main,
         ylab = "Total Change in Location (mm)",
         las = 2,
         border = NA)

  abline(h = 0, lty = 2, col = "gray40", lwd = 2)
  grid(nx = NA, ny = NULL, col = "gray90", lty = 1)
}

plot_comparison_return_levels <- function(stat_results, nonstat_results,
                                         return_period = 100,
                                         target_year = NULL) {

  target_years <- attr(nonstat_results, "target_years")

  if (is.null(target_year)) {
    target_year <- max(target_years)
  }

  # Get return levels
  stat_col <- paste0("RL", return_period)
  nonstat_col <- sprintf("RL%d_Year%d", return_period, target_year)

  if (!stat_col %in% names(stat_results) || !nonstat_col %in% names(nonstat_results)) {
    cat("Return level columns not found\n")
    return(invisible(NULL))
  }

  stat_rl <- stat_results[[stat_col]]
  nonstat_rl <- nonstat_results[[nonstat_col]]

  # Create grouped barplot
  plot_data <- rbind(stat_rl, nonstat_rl)
  colnames(plot_data) <- stat_results$Duration

  barplot(plot_data,
         beside = TRUE,
         col = c("steelblue", "darkred"),
         main = sprintf("%d-Year Return Levels: Stationary vs Nonstationary (Year %d)",
                       return_period, target_year),
         ylab = "Precipitation (mm)",
         las = 2,
         border = NA)

  legend("topleft",
        legend = c("Stationary", sprintf("Nonstationary (Year %d)", target_year)),
        fill = c("steelblue", "darkred"),
        bty = "n")

  grid(nx = NA, ny = NULL, col = "gray90", lty = 1)
}

plot_nonstationary_evolution <- function(nonstat_results, duration_name,
                                        return_period = 100) {

  target_years <- attr(nonstat_results, "target_years")

  # Get return levels for all target years
  rl_values <- numeric(length(target_years))
  for (i in seq_along(target_years)) {
    col_name <- sprintf("RL%d_Year%d", return_period, target_years[i])
    idx <- which(nonstat_results$Duration == duration_name)
    if (length(idx) > 0) {
      rl_values[i] <- nonstat_results[[col_name]][idx]
    }
  }

  plot(target_years, rl_values,
      type = "b", pch = 19, col = "darkred", lwd = 2,
      xlab = "Year",
      ylab = sprintf("RL%d (mm)", return_period),
      main = sprintf("%d-Year Return Level Evolution: %s",
                    return_period, duration_name),
      las = 1)

  grid(col = "gray90", lty = 1)

  # Add trend line if there's a trend
  idx <- which(nonstat_results$Duration == duration_name)
  if (length(idx) > 0 && !is.na(nonstat_results$Location_mu1[idx])) {
    trend <- nonstat_results$Location_mu1[idx]
    total_change <- nonstat_results$Total_Change[idx]

    text(mean(target_years), max(rl_values, na.rm = TRUE) * 0.95,
        labels = sprintf("Trend: %.4f mm/year\nTotal change: %.2f mm",
                        trend, total_change),
        pos = 1, col = "darkred")
  }
}

plot_idf_nonstationary <- function(nonstat_results,
                                  return_periods = c(2, 10, 25, 50, 100),
                                  target_years = NULL,
                                  colors = NULL,
                                  log_scale = TRUE) {

  if (is.null(target_years)) {
    target_years <- attr(nonstat_results, "target_years")
  }

  if (is.null(colors)) {
    colors <- c("darkgreen", "blue", "orange", "red", "darkred")
  }

  durations <- DURATIONS[match(nonstat_results$Duration, DURATION_NAMES)]

  # Use the last year for IDF curves
  target_year <- max(target_years)

  log_axis <- if (log_scale) "xy" else ""

  # Get all return level columns for target year
  rl_cols <- sprintf("RL%d_Year%d", return_periods, target_year)
  plot_data <- nonstat_results[, rl_cols, drop = FALSE]

  plot(1, type = "n",
      xlim = range(durations),
      ylim = range(plot_data, na.rm = TRUE),
      log = log_axis,
      xlab = "Duration (minutes)",
      ylab = "Precipitation (mm)",
      main = sprintf("IDF Curves - Nonstationary (Year %d)", target_year),
      las = 1)

  for (i in seq_along(return_periods)) {
    rp <- return_periods[i]
    col_name <- sprintf("RL%d_Year%d", rp, target_year)

    if (col_name %in% names(nonstat_results)) {
      lines(durations, nonstat_results[[col_name]],
           type = "b", pch = 19, col = colors[i], lwd = 2)
    }
  }

  legend("topleft",
        legend = paste0(return_periods, "-year"),
        col = colors[1:length(return_periods)],
        lwd = 2,
        pch = 19,
        bty = "n",
        title = "Return Period")

  grid(col = "gray80", lty = 1)
}

plot_nonstationary_summary <- function(stat_results, nonstat_results,
                                      annual_max_df = NULL,
                                      return_period = 100) {

  par(mfrow = c(2, 3), mar = c(5, 4, 3, 2))

  target_years <- attr(nonstat_results, "target_years")

  # Plot 1: Stationary return levels
  plot_return_levels(stat_results, return_period = return_period,
                    main = sprintf("Stationary: RL%d", return_period))

  # Plot 2: Nonstationary return levels (last year)
  target_year <- max(target_years)
  nonstat_col <- sprintf("RL%d_Year%d", return_period, target_year)
  if (nonstat_col %in% names(nonstat_results)) {
    barplot(nonstat_results[[nonstat_col]],
           names.arg = nonstat_results$Duration,
           col = "darkred",
           main = sprintf("Nonstationary: RL%d (Year %d)", return_period, target_year),
           ylab = "Precipitation (mm)",
           las = 2,
           border = NA)
    grid(nx = NA, ny = NULL, col = "gray90", lty = 1)
  }

  # Plot 3: Trend in location
  plot_nonstationary_trends(nonstat_results)

  # Plot 4: Comparison
  plot_comparison_return_levels(stat_results, nonstat_results,
                               return_period = return_period,
                               target_year = target_year)

  # Plot 5: Stationary IDF
  plot_idf_curve(stat_results, log_scale = TRUE)

  # Plot 6: Nonstationary IDF
  plot_idf_nonstationary(nonstat_results, target_years = target_year, log_scale = TRUE)

  par(mfrow = c(1, 1))
}

export_nonstationary_plots_pdf <- function(stat_results, nonstat_results,
                                          annual_max_df = NULL,
                                          output_path = "nonstationary_comparison.pdf",
                                          return_period = 100,
                                          verbose = TRUE) {

  pdf(output_path, width = 14, height = 8.5)

  # Summary comparison
  plot_nonstationary_summary(stat_results, nonstat_results,
                            annual_max_df, return_period)

  # Individual duration evolution plots
  par(mfrow = c(3, 3), mar = c(4, 4, 3, 2))
  for (dur_name in nonstat_results$Duration) {
    plot_nonstationary_evolution(nonstat_results, dur_name, return_period)
  }
  par(mfrow = c(1, 1))

  # IDF comparison at different time periods
  target_years <- attr(nonstat_results, "target_years")
  par(mfrow = c(2, 2), mar = c(5, 5, 4, 2))

  plot_idf_curve(stat_results, main = "IDF: Stationary")

  for (year in target_years) {
    plot_idf_nonstationary(nonstat_results, target_years = year,
                          main = sprintf("IDF: Nonstationary (Year %d)", year))
  }
  par(mfrow = c(1, 1))

  dev.off()

  if (verbose) {
    cat(sprintf("\n✓ Nonstationary comparison plots exported to: %s\n", output_path))
  }
}

create_summary_table <- function(results, annual_max_df, return_period = 100) {
  rl_col <- paste0("RL", return_period)

  obs_max <- sapply(results$Duration, function(dur) {
    if (dur %in% names(annual_max_df)) {
      return(safe_max(annual_max_df[[dur]]))
    } else {
      return(NA)
    }
  })

  summary_table <- data.frame(
    Duration = results$Duration,
    Mean = round(results$Mean, 1),
    SD = round(results$SD, 1),
    Max = round(obs_max, 1),
    Location = round(results$Location, 2),
    Scale = round(results$Scale, 2)
  )

  summary_table[[paste0("RL", return_period)]] <- round(results[[rl_col]], 1)

  return(summary_table)
}

################################################################################
## MAIN ANALYSIS WORKFLOW
################################################################################

cat("\n")
cat("################################################################################\n")
cat("## GUMBEL EXTREME RAINFALL ANALYSIS - STARTING\n")
cat("################################################################################\n")

# Create output directory
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
  cat(sprintf("\n✓ Created output directory: %s\n", OUTPUT_DIR))
}

# Step 1: Load precipitation data
yearly_data <- load_precipitation_data(
  data_dir = DATA_DIR,
  start_year = START_YEAR,
  end_year = END_YEAR,
  scenario = SCENARIO,
  station_id = STATION_ID,
  verbose = VERBOSE
)

# Step 2: Calculate annual maxima
annual_max <- calculate_annual_maxima(
  df_long = yearly_data,
  durations = DURATIONS,
  duration_names = DURATION_NAMES,
  verbose = VERBOSE
)

# Export annual maxima
if (EXPORT_ANNUAL_MAX) {
  annual_max_path <- file.path(OUTPUT_DIR, "annual_maxima.csv")
  write.csv(annual_max, annual_max_path, row.names = FALSE)
  cat(sprintf("\n✓ Annual maxima exported to: %s\n", annual_max_path))
}

# Step 3: Fit Gumbel distributions and calculate return levels
results <- analyze_all_durations(
  annual_max_df = annual_max,
  durations = DURATIONS,
  duration_names = DURATION_NAMES,
  return_periods = RETURN_PERIODS,
  method = "MLE",
  verbose = VERBOSE
)

# Step 4: Quality checks
quality_checks <- check_return_levels(
  results = results,
  return_period = 100,
  verbose = VERBOSE
)

# Step 5: Create summary table
print_header("SUMMARY TABLE")

summary_table <- create_summary_table(
  results = results,
  annual_max_df = annual_max,
  return_period = 100
)

print(summary_table)

# Step 6: Create visualizations
print_header("CREATING VISUALIZATIONS")

cat("\nGenerating summary plots...\n")
plot_summary(results, annual_max, return_period = 100)

# Step 7: Export results
print_header("EXPORTING RESULTS")

if (EXPORT_RESULTS) {
  results_clean <- results
  attr(results_clean, "fitted_models") <- NULL

  results_path <- file.path(OUTPUT_DIR, "stationary_gumbel_results.csv")
  write.csv(results_clean, results_path, row.names = FALSE)
  cat(sprintf("✓ Gumbel results exported to: %s\n", results_path))
}

if (EXPORT_PLOTS) {
  plots_path <- file.path(OUTPUT_DIR, "gumbel_analysis_plots.pdf")
  export_plots_pdf(
    results = results,
    annual_max_df = annual_max,
    output_path = plots_path,
    return_period = 100,
    verbose = TRUE
  )
}

################################################################################
## NONSTATIONARY ANALYSIS (Optional)
################################################################################

if (RUN_NONSTATIONARY) {
  print_header("NONSTATIONARY GUMBEL ANALYSIS")

  cat("\nStarting nonstationary analysis with time-varying parameters...\n")

  # Step 8: Fit nonstationary Gumbel distributions
  nonstat_results <- analyze_all_durations_nonstationary(
    annual_max_df = annual_max,
    durations = DURATIONS,
    duration_names = DURATION_NAMES,
    return_periods = RETURN_PERIODS,
    trend_location = NONSTAT_TREND_LOCATION,
    trend_scale = NONSTAT_TREND_SCALE,
    method = "MLE",
    verbose = VERBOSE
  )

  # Step 9: Compare stationary vs nonstationary
  comparison <- compare_stationary_nonstationary(
    stat_results = results,
    nonstat_results = nonstat_results,
    annual_max_df = annual_max,
    verbose = VERBOSE
  )

  # Step 10: Create nonstationary visualizations
  print_header("NONSTATIONARY VISUALIZATIONS")

  cat("\nGenerating nonstationary comparison plots...\n")
  plot_nonstationary_summary(results, nonstat_results, annual_max, return_period = 100)

  # Step 11: Export nonstationary results
  print_header("EXPORTING NONSTATIONARY RESULTS")

  if (EXPORT_RESULTS) {
    nonstat_results_clean <- nonstat_results
    attr(nonstat_results_clean, "fitted_models") <- NULL
    attr(nonstat_results_clean, "target_years") <- NULL
    attr(nonstat_results_clean, "trend_location") <- NULL
    attr(nonstat_results_clean, "trend_scale") <- NULL

    nonstat_path <- file.path(OUTPUT_DIR, "nonstationary_gumbel_results.csv")
    write.csv(nonstat_results_clean, nonstat_path, row.names = FALSE)
    cat(sprintf("✓ Nonstationary results exported to: %s\n", nonstat_path))

    comparison_path <- file.path(OUTPUT_DIR, "model_comparison.csv")
    write.csv(comparison, comparison_path, row.names = FALSE)
    cat(sprintf("✓ Model comparison exported to: %s\n", comparison_path))
  }

  if (EXPORT_PLOTS) {
    nonstat_plots_path <- file.path(OUTPUT_DIR, "nonstationary_comparison.pdf")
    export_nonstationary_plots_pdf(
      stat_results = results,
      nonstat_results = nonstat_results,
      annual_max_df = annual_max,
      output_path = nonstat_plots_path,
      return_period = 100,
      verbose = TRUE
    )
  }
}

# Final summary
print_header("ANALYSIS COMPLETE")

cat("\nOutput files:\n")
cat("\nStationary Analysis:\n")
if (EXPORT_ANNUAL_MAX) {
  cat(sprintf("  - Annual maxima: %s\n",
             file.path(OUTPUT_DIR, "annual_maxima.csv")))
}
if (EXPORT_RESULTS) {
  cat(sprintf("  - Gumbel results: %s\n",
             file.path(OUTPUT_DIR, "stationary_gumbel_results.csv")))
}
if (EXPORT_PLOTS) {
  cat(sprintf("  - Plots: %s\n",
             file.path(OUTPUT_DIR, "gumbel_analysis_plots.pdf")))
}

if (RUN_NONSTATIONARY) {
  cat("\nNonstationary Analysis:\n")
  if (EXPORT_RESULTS) {
    cat(sprintf("  - Nonstationary results: %s\n",
               file.path(OUTPUT_DIR, "nonstationary_gumbel_results.csv")))
    cat(sprintf("  - Model comparison: %s\n",
               file.path(OUTPUT_DIR, "model_comparison.csv")))
  }
  if (EXPORT_PLOTS) {
    cat(sprintf("  - Comparison plots: %s\n",
               file.path(OUTPUT_DIR, "nonstationary_comparison.pdf")))
  }
}

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("Analysis completed successfully!\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("\n")
