################################################################################
## SIMPLE EXAMPLE - GUMBEL ANALYSIS
## Minimal example showing the basic workflow
################################################################################

# Install required package if needed
# install.packages("extRemes")

library(extRemes)

# Load modular functions
source("R/utils.R")
source("R/data_loading.R")
source("R/annual_maxima.R")
source("R/gumbel_analysis.R")
source("R/visualization.R")

# Load configuration (edit config.R to customize)
source("config.R")

################################################################################
## QUICK START - 3 STEPS
################################################################################

# Step 1: Load your precipitation data
cat("\n=== STEP 1: Loading data ===\n")
yearly_data <- load_precipitation_data(
  data_dir = CONFIG$data_dir,
  start_year = CONFIG$start_year,
  end_year = CONFIG$end_year,
  scenario = CONFIG$scenario,
  station_id = CONFIG$station_id
)

# Step 2: Calculate annual maxima and fit Gumbel distributions
cat("\n=== STEP 2: Calculating annual maxima ===\n")
annual_max <- calculate_annual_maxima(yearly_data)

cat("\n=== STEP 3: Fitting Gumbel distributions ===\n")
results <- analyze_all_durations(annual_max)

# Step 3: View results and create plots
cat("\n=== RESULTS ===\n")
print(results[, c("Duration", "Mean", "Location", "Scale", "RL100")])

cat("\n=== Creating plots ===\n")
plot_summary(results, annual_max)

# Export results
cat("\n=== Exporting results ===\n")
if (!dir.exists(CONFIG$output_dir)) {
  dir.create(CONFIG$output_dir)
}

write.csv(results, file.path(CONFIG$output_dir, "results.csv"), row.names = FALSE)
export_plots_pdf(results, annual_max, file.path(CONFIG$output_dir, "plots.pdf"))

cat("\n✓ Analysis complete! Check the output/ folder for results.\n")
