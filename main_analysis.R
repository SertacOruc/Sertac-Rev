################################################################################
## STATIONARY GUMBEL ANALYSIS - MAIN SCRIPT
## Modular implementation of extreme rainfall analysis
##
## This script demonstrates the complete workflow:
##   1. Load precipitation data
##   2. Calculate annual maxima for multiple durations
##   3. Fit Gumbel distributions
##   4. Calculate return levels
##   5. Create visualizations
##   6. Export results
################################################################################

# Load required libraries
if (!requireNamespace("extRemes", quietly = TRUE)) {
  stop("Please install extRemes: install.packages('extRemes')")
}

library(extRemes)

# Source modular functions
source("R/utils.R")
source("R/data_loading.R")
source("R/annual_maxima.R")
source("R/gumbel_analysis.R")
source("R/visualization.R")

################################################################################
## CONFIGURATION
################################################################################

# Data parameters
DATA_DIR <- "C:/Users/ser_o/OneDrive/Documents/EXTREMES/hyetos/hyetos"
START_YEAR <- 2015
END_YEAR <- 2100
SCENARIO <- "ssp245"
STATION_ID <- 31

# Analysis parameters (using defaults from utils.R)
# DURATIONS, DURATION_NAMES, RETURN_PERIODS

# Output parameters
OUTPUT_DIR <- "output"
EXPORT_ANNUAL_MAX <- TRUE
EXPORT_RESULTS <- TRUE
EXPORT_PLOTS <- TRUE

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

################################################################################
## STEP 1: LOAD DATA
################################################################################

cat("\n")
cat("################################################################################\n")
cat("## GUMBEL EXTREME RAINFALL ANALYSIS\n")
cat("################################################################################\n")

# Option 1: Load from hyetos files
yearly_data <- load_precipitation_data(
  data_dir = DATA_DIR,
  start_year = START_YEAR,
  end_year = END_YEAR,
  scenario = SCENARIO,
  station_id = STATION_ID,
  verbose = TRUE
)

# Option 2: Load from CSV (if data already processed)
# yearly_data <- load_precipitation_csv("data/precipitation_data.csv", verbose = TRUE)

################################################################################
## STEP 2: CALCULATE ANNUAL MAXIMA
################################################################################

annual_max <- calculate_annual_maxima(
  df_long = yearly_data,
  durations = DURATIONS,
  duration_names = DURATION_NAMES,
  verbose = TRUE
)

# Export annual maxima
if (EXPORT_ANNUAL_MAX) {
  export_annual_maxima(
    annual_max_df = annual_max,
    output_path = file.path(OUTPUT_DIR, "annual_maxima.csv"),
    verbose = TRUE
  )
}

################################################################################
## STEP 3: FIT GUMBEL DISTRIBUTIONS AND CALCULATE RETURN LEVELS
################################################################################

results <- analyze_all_durations(
  annual_max_df = annual_max,
  durations = DURATIONS,
  duration_names = DURATION_NAMES,
  return_periods = RETURN_PERIODS,
  method = "MLE",
  verbose = TRUE
)

################################################################################
## STEP 4: QUALITY CHECKS
################################################################################

quality_checks <- check_return_levels(
  results = results,
  return_period = 100,
  verbose = TRUE
)

################################################################################
## STEP 5: CREATE SUMMARY TABLE
################################################################################

print_header("SUMMARY TABLE")

summary_table <- create_summary_table(
  results = results,
  annual_max_df = annual_max,
  return_period = 100
)

print(summary_table)

################################################################################
## STEP 6: VISUALIZATIONS
################################################################################

print_header("CREATING VISUALIZATIONS")

# Summary plots
cat("\nGenerating summary plots...\n")
plot_summary(results, annual_max, return_period = 100)

# Wait for user to view plot
readline(prompt = "Press [Enter] to continue to next plot...")

# IDF curves
cat("\nGenerating IDF curves...\n")
par(mfrow = c(1, 1))
plot_idf_curve(results, return_periods = c(2, 5, 10, 25, 50, 100))

readline(prompt = "Press [Enter] to continue...")

# Time series for selected durations
cat("\nGenerating time series plots...\n")
par(mfrow = c(2, 2))
for (dur in c("5min", "1hr", "6hr", "24hr")) {
  plot_annual_maxima_ts(annual_max, dur)
}
par(mfrow = c(1, 1))

################################################################################
## STEP 7: EXPORT RESULTS
################################################################################

print_header("EXPORTING RESULTS")

# Export results table
if (EXPORT_RESULTS) {
  export_gumbel_results(
    results = results,
    output_path = file.path(OUTPUT_DIR, "stationary_gumbel_results.csv"),
    verbose = TRUE
  )
}

# Export plots to PDF
if (EXPORT_PLOTS) {
  export_plots_pdf(
    results = results,
    annual_max_df = annual_max,
    output_path = file.path(OUTPUT_DIR, "gumbel_analysis_plots.pdf"),
    return_period = 100,
    verbose = TRUE
  )
}

################################################################################
## ANALYSIS COMPLETE
################################################################################

print_header("ANALYSIS COMPLETE")

cat("\nOutput files:\n")
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

cat("\nSession info:\n")
cat(sprintf("  R version: %s\n", R.version.string))
cat(sprintf("  extRemes version: %s\n", packageVersion("extRemes")))

cat("\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("Analysis completed successfully!\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("\n")
