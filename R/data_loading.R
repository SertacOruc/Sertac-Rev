################################################################################
## DATA LOADING FUNCTIONS
## Functions for loading and preparing hyetos precipitation data
################################################################################

source("R/utils.R")

#' Load hyetos precipitation data for a single month
#' @param data_dir Directory containing hyetos output files
#' @param month Month number (1-12)
#' @param scenario Climate scenario (e.g., "ssp245")
#' @param station_id Station identifier
#' @param n_intervals Number of intervals (columns, default 288)
#' @return Matrix of precipitation data
#' @export
load_hyetos_month <- function(data_dir, month, scenario = "ssp245",
                               station_id = 31, n_intervals = 288) {

  # Construct filename
  filename <- sprintf("hyetosout_%s_ %d _ %d _ %d .txt",
                     scenario, station_id, month, station_id)
  filepath <- file.path(data_dir, filename)

  # Check file exists
  if (!file.exists(filepath)) {
    stop(sprintf("File not found: %s", filepath))
  }

  # Read data (columns 5-292 contain the precipitation data)
  tryCatch({
    data_matrix <- as.matrix(read.table(filepath))[, 5:(4 + n_intervals)]
    return(data_matrix)
  }, error = function(e) {
    stop(sprintf("Error reading file %s: %s", filepath, e$message))
  })
}

#' Load and prepare complete precipitation dataset
#' @param data_dir Directory containing hyetos output files
#' @param start_year First year of analysis
#' @param end_year Last year of analysis
#' @param scenario Climate scenario
#' @param station_id Station identifier
#' @param verbose Print progress messages
#' @return Data frame with columns: date, year, interval, precip
#' @export
load_precipitation_data <- function(data_dir,
                                   start_year = 2015,
                                   end_year = 2100,
                                   scenario = "ssp245",
                                   station_id = 31,
                                   verbose = TRUE) {

  if (verbose) {
    print_header("LOADING PRECIPITATION DATA")
  }

  # Calculate dimensions
  n_years <- end_year - start_year + 1
  n_months <- MONTHS_PER_YEAR
  total_records <- n_years * DAYS_PER_MONTH * n_months * INTERVALS_PER_DAY

  # Pre-allocate result array
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

      # Load month data
      month_data <- load_hyetos_month(
        data_dir = data_dir,
        month = month,
        scenario = scenario,
        station_id = station_id
      )

      # Prepare monthly container (handle variable month lengths)
      n_days_in_file <- nrow(month_data)
      data_filled <- matrix(0, nrow = DAYS_PER_MONTH * n_years, ncol = INTERVALS_PER_DAY)

      if (n_days_in_file > 0) {
        data_filled[1:n_days_in_file, ] <- month_data
      }

      # Extract this year's data
      year_start_row <- (year_idx - 1) * DAYS_PER_MONTH + 1
      year_end_row <- year_idx * DAYS_PER_MONTH
      year_month_data <- data_filled[year_start_row:year_end_row, ]

      # Fill into result dataframe
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

  # Create calendar
  calendar_vec <- create_360day_calendar(start_year, end_year, INTERVALS_PER_DAY)
  yearly_data$date <- calendar_vec

  # Handle NA values
  yearly_data$precip[is.na(yearly_data$precip)] <- 0

  # Validate
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

#' Load precipitation data from CSV file (alternative to hyetos)
#' @param csv_path Path to CSV file
#' @param verbose Print progress messages
#' @return Data frame with columns: date, year, interval, precip
#' @export
load_precipitation_csv <- function(csv_path, verbose = TRUE) {

  if (!file.exists(csv_path)) {
    stop(sprintf("CSV file not found: %s", csv_path))
  }

  if (verbose) {
    print_header("LOADING PRECIPITATION DATA FROM CSV")
    cat(sprintf("Reading: %s\n", csv_path))
  }

  data <- read.csv(csv_path, stringsAsFactors = FALSE)

  # Validate required columns
  required_cols <- c("date", "year", "interval", "precip")
  missing_cols <- setdiff(required_cols, names(data))

  if (length(missing_cols) > 0) {
    stop(sprintf("Missing required columns: %s", paste(missing_cols, collapse = ", ")))
  }

  # Convert date if needed
  if (!inherits(data$date, "Date")) {
    data$date <- as.Date(data$date)
  }

  # Validate
  validate_precip_data(data$precip)
  validate_years(unique(data$year))

  if (verbose) {
    cat(sprintf("\nData loaded successfully:\n"))
    cat(sprintf("  Records: %d\n", nrow(data)))
    cat(sprintf("  Years: %d - %d\n", min(data$year), max(data$year)))
    cat(sprintf("  Total precipitation: %.2f mm\n", sum(data$precip, na.rm = TRUE)))
  }

  return(data)
}
