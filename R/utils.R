################################################################################
## UTILITY FUNCTIONS AND CONSTANTS
## Shared constants and helper functions for extreme rainfall analysis
################################################################################

#' Global constants for duration analysis
#' @export
DURATIONS <- c(5, 10, 15, 30, 60, 180, 360, 720, 1440)
DURATION_NAMES <- c("5min", "10min", "15min", "30min", "1hr", "3hr", "6hr", "12hr", "24hr")
RETURN_PERIODS <- c(2, 5, 10, 25, 50, 100)

# Analysis parameters
INTERVALS_PER_DAY <- 288  # 5-minute intervals
DAYS_PER_MONTH <- 30      # 360-day year
MONTHS_PER_YEAR <- 12
INTERVAL_MINUTES <- 5

#' Validate duration index
#' @param duration_idx Index of duration (1-9)
#' @return TRUE if valid, stops with error if invalid
#' @export
validate_duration_index <- function(duration_idx) {
  if (!is.numeric(duration_idx) || duration_idx < 1 || duration_idx > length(DURATIONS)) {
    stop(sprintf("Invalid duration index: %s. Must be between 1 and %d",
                 duration_idx, length(DURATIONS)))
  }
  return(TRUE)
}

#' Validate year range
#' @param years Vector of years
#' @param min_years Minimum number of years required
#' @return TRUE if valid, stops with error if invalid
#' @export
validate_years <- function(years, min_years = 10) {
  if (!is.numeric(years) || length(years) < min_years) {
    stop(sprintf("Insufficient years: %d. Need at least %d years for robust analysis",
                 length(years), min_years))
  }
  if (any(is.na(years))) {
    stop("Years contain NA values")
  }
  return(TRUE)
}

#' Validate precipitation data
#' @param data Precipitation vector or matrix
#' @return TRUE if valid, stops with error if invalid
#' @export
validate_precip_data <- function(data) {
  if (!is.numeric(data)) {
    stop("Precipitation data must be numeric")
  }
  if (any(data < 0, na.rm = TRUE)) {
    warning("Negative precipitation values detected. Setting to 0.")
  }
  return(TRUE)
}

#' Create calendar for 360-day years
#' @param start_year First year
#' @param end_year Last year
#' @param intervals_per_day Number of intervals per day
#' @return Vector of dates
#' @export
create_360day_calendar <- function(start_year, end_year, intervals_per_day = 288) {

  years <- start_year:end_year
  calendar_vec <- c()

  for (yy in years) {
    # 360-day year starting on Jan 1
    days_360 <- as.Date(paste0(yy, "-01-01")) + 0:359

    # Repeat each day for intervals
    days_360_rep <- rep(days_360, each = intervals_per_day)

    # Add to main vector
    calendar_vec <- c(calendar_vec, days_360_rep)
  }

  return(calendar_vec)
}

#' Print analysis header
#' @param title Header title
#' @export
print_header <- function(title) {
  cat("\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
  cat(paste0("  ", title, "\n"))
  cat(paste(rep("=", 80), collapse = ""), "\n")
}

#' Print section separator
#' @param title Section title
#' @export
print_section <- function(title) {
  cat(sprintf("\n--- %s ---\n", title))
}

#' Safe maximum calculation (handles NA and empty vectors)
#' @param x Numeric vector
#' @return Maximum value or NA
#' @export
safe_max <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA)
  }
  return(max(x, na.rm = TRUE))
}

#' Safe mean calculation
#' @param x Numeric vector
#' @return Mean value or NA
#' @export
safe_mean <- function(x) {
  if (length(x) == 0 || all(is.na(x))) {
    return(NA)
  }
  return(mean(x, na.rm = TRUE))
}

#' Check monotonicity of values
#' @param values Numeric vector
#' @param names Names for values
#' @param tolerance Relative tolerance for near-equality
#' @return List with is_monotonic flag and violations
#' @export
check_monotonic <- function(values, names = NULL, tolerance = 1e-6) {

  if (is.null(names)) {
    names <- paste0("Val", seq_along(values))
  }

  violations <- list()
  is_monotonic <- TRUE

  for (i in 1:(length(values) - 1)) {
    if (!is.na(values[i]) && !is.na(values[i + 1])) {
      # Check if current value is greater than next (allowing for tolerance)
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

  return(list(
    is_monotonic = is_monotonic,
    violations = violations
  ))
}
