################################################################################
## CONFIGURATION FILE
## Edit this file to customize your analysis parameters
################################################################################

# ==============================================================================
# DATA CONFIGURATION
# ==============================================================================

# Directory containing hyetos output files
# Update this to your data location
CONFIG <- list(

  # Data source
  data_dir = "C:/Users/ser_o/OneDrive/Documents/EXTREMES/hyetos/hyetos",

  # Time range
  start_year = 2015,
  end_year = 2100,

  # Scenario and station
  scenario = "ssp245",
  station_id = 31,

  # ==============================================================================
  # ANALYSIS CONFIGURATION
  # ==============================================================================

  # Durations to analyze (in minutes)
  # Default: c(5, 10, 15, 30, 60, 180, 360, 720, 1440)
  durations = c(5, 10, 15, 30, 60, 180, 360, 720, 1440),

  # Duration names for display
  # Default: c("5min", "10min", "15min", "30min", "1hr", "3hr", "6hr", "12hr", "24hr")
  duration_names = c("5min", "10min", "15min", "30min", "1hr", "3hr", "6hr", "12hr", "24hr"),

  # Return periods to calculate (in years)
  # Default: c(2, 5, 10, 25, 50, 100)
  return_periods = c(2, 5, 10, 25, 50, 100),

  # Fitting method: "MLE" (Maximum Likelihood) or "GMLE" (Generalized MLE)
  fitting_method = "MLE",

  # ==============================================================================
  # OUTPUT CONFIGURATION
  # ==============================================================================

  # Output directory
  output_dir = "output",

  # Export options
  export_annual_max = TRUE,
  export_results = TRUE,
  export_plots = TRUE,

  # Verbosity (print detailed progress)
  verbose = TRUE
)

# ==============================================================================
# CUSTOM DURATIONS (EXAMPLE)
# ==============================================================================
# Uncomment and modify if you want to analyze different durations

# CONFIG$durations <- c(10, 30, 60, 120, 360, 720, 1440)
# CONFIG$duration_names <- c("10min", "30min", "1hr", "2hr", "6hr", "12hr", "24hr")

# ==============================================================================
# VALIDATION
# ==============================================================================

# Check that durations and names match
if (length(CONFIG$durations) != length(CONFIG$duration_names)) {
  stop("Error: durations and duration_names must have the same length")
}

# Check that return periods are positive
if (any(CONFIG$return_periods <= 0)) {
  stop("Error: return_periods must be positive")
}

cat("Configuration loaded successfully\n")
cat(sprintf("  Data: %s\n", CONFIG$data_dir))
cat(sprintf("  Years: %d - %d\n", CONFIG$start_year, CONFIG$end_year))
cat(sprintf("  Durations: %d\n", length(CONFIG$durations)))
cat(sprintf("  Return periods: %s\n", paste(CONFIG$return_periods, collapse = ", ")))
