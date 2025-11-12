################################################################################
## VISUALIZATION FUNCTIONS
## Functions for creating plots and charts for extreme rainfall analysis
################################################################################

source("R/utils.R")

#' Plot return levels comparison
#' @param results Data frame from analyze_all_durations
#' @param return_period Return period to plot
#' @param col Bar color
#' @param main Plot title (auto-generated if NULL)
#' @export
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

#' Plot scale parameters
#' @param results Data frame from analyze_all_durations
#' @param col Bar color
#' @export
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

#' Plot mean vs return level comparison
#' @param results Data frame from analyze_all_durations
#' @param return_period Return period to compare
#' @export
plot_mean_vs_return_level <- function(results, return_period = 100) {

  rl_col <- paste0("RL", return_period)

  plot(results$Mean, results[[rl_col]],
      pch = 19, col = "darkblue", cex = 1.5,
      xlab = "Mean Annual Maximum (mm)",
      ylab = sprintf("RL%d (mm)", return_period),
      main = sprintf("Mean vs RL%d", return_period))

  # Add labels
  text(results$Mean, results[[rl_col]],
      labels = results$Duration,
      pos = 3, cex = 0.8, col = "darkblue")

  # Add 1:1 reference line
  abline(0, 1, col = "red", lty = 2, lwd = 2)

  # Add legend
  legend("topleft",
        legend = c("Data", "1:1 Line"),
        col = c("darkblue", "red"),
        pch = c(19, NA),
        lty = c(NA, 2),
        lwd = c(NA, 2),
        bty = "n")

  grid(col = "gray90", lty = 1)
}

#' Plot IDF curve (Intensity-Duration-Frequency)
#' @param results Data frame from analyze_all_durations
#' @param return_periods Vector of return periods to plot
#' @param colors Vector of colors for each return period
#' @param log_scale Use log-log scale (default TRUE)
#' @export
plot_idf_curve <- function(results,
                          return_periods = c(2, 10, 25, 50, 100),
                          colors = NULL,
                          log_scale = TRUE) {

  if (is.null(colors)) {
    colors <- c("darkgreen", "blue", "orange", "red", "darkred")
  }

  # Get duration values
  durations <- DURATIONS[match(results$Duration, DURATION_NAMES)]

  # Set up plot
  plot_type <- if (log_scale) "n" else "n"
  log_axis <- if (log_scale) "xy" else ""

  plot(1, type = "n",
      xlim = range(durations),
      ylim = range(results[, grep("^RL", names(results))], na.rm = TRUE),
      log = log_axis,
      xlab = "Duration (minutes)",
      ylab = "Precipitation (mm)",
      main = "IDF Curves - Gumbel Analysis",
      las = 1)

  # Plot each return period
  for (i in seq_along(return_periods)) {
    rp <- return_periods[i]
    rl_col <- paste0("RL", rp)

    if (rl_col %in% names(results)) {
      lines(durations, results[[rl_col]],
           type = "b", pch = 19, col = colors[i], lwd = 2)
    }
  }

  # Add legend
  legend("topleft",
        legend = paste0(return_periods, "-year"),
        col = colors[1:length(return_periods)],
        lwd = 2,
        pch = 19,
        bty = "n",
        title = "Return Period")

  grid(col = "gray80", lty = 1)
}

#' Plot annual maxima time series for specific duration
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param duration_name Duration to plot
#' @param col Line color
#' @export
plot_annual_maxima_ts <- function(annual_max_df, duration_name = "1hr",
                                 col = "darkblue") {

  if (!duration_name %in% names(annual_max_df)) {
    stop(sprintf("Duration '%s' not found", duration_name))
  }

  years <- annual_max_df$year
  values <- annual_max_df[[duration_name]]

  plot(years, values,
      type = "l", lwd = 2, col = col,
      xlab = "Year",
      ylab = "Annual Maximum (mm)",
      main = sprintf("Annual Maximum Time Series - %s", duration_name))

  # Add points
  points(years, values, pch = 19, col = col, cex = 0.8)

  # Add trend line
  trend <- lm(values ~ years)
  abline(trend, col = "red", lty = 2, lwd = 2)

  # Add mean line
  abline(h = mean(values, na.rm = TRUE), col = "gray50", lty = 3, lwd = 1.5)

  # Add legend
  legend("topleft",
        legend = c("Annual Max", "Trend", "Mean"),
        col = c(col, "red", "gray50"),
        lty = c(1, 2, 3),
        lwd = c(2, 2, 1.5),
        bty = "n")

  grid(col = "gray90", lty = 1)
}

#' Create comprehensive summary plot
#' @param results Data frame from analyze_all_durations
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param return_period Primary return period for plotting
#' @export
plot_summary <- function(results, annual_max_df = NULL, return_period = 100) {

  # Set up 2x2 layout
  par(mfrow = c(2, 2), mar = c(5, 4, 3, 2))

  # 1. Return levels
  plot_return_levels(results, return_period = return_period)

  # 2. Scale parameters
  plot_scale_parameters(results)

  # 3. Mean vs Return Level
  plot_mean_vs_return_level(results, return_period = return_period)

  # 4. IDF curve
  plot_idf_curve(results, log_scale = TRUE)

  # Reset layout
  par(mfrow = c(1, 1))
}

#' Create detailed diagnostic plots
#' @param results Data frame from analyze_all_durations
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param duration_name Duration to diagnose
#' @export
plot_diagnostics <- function(results, annual_max_df, duration_name = "1hr") {

  if (!duration_name %in% names(annual_max_df)) {
    stop(sprintf("Duration '%s' not found", duration_name))
  }

  # Get fitted model
  fitted_models <- attr(results, "fitted_models")

  if (is.null(fitted_models) || !duration_name %in% names(fitted_models)) {
    stop("Fitted models not available. Run analyze_all_durations first.")
  }

  fit <- fitted_models[[duration_name]]

  if (is.null(fit)) {
    stop(sprintf("No fitted model for %s", duration_name))
  }

  # Set up 2x2 layout
  par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

  # Use extRemes plot function
  plot(fit, main = sprintf("Diagnostics: %s", duration_name))

  # Reset layout
  par(mfrow = c(1, 1))
}

#' Export all plots to PDF
#' @param results Data frame from analyze_all_durations
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param output_path Output PDF path
#' @param return_period Primary return period
#' @param verbose Print confirmation
#' @export
export_plots_pdf <- function(results, annual_max_df = NULL,
                            output_path = "gumbel_analysis_plots.pdf",
                            return_period = 100,
                            verbose = TRUE) {

  pdf(output_path, width = 11, height = 8.5)

  # Page 1: Summary plots
  plot_summary(results, annual_max_df, return_period)

  # Page 2: IDF curves (separate, larger)
  par(mfrow = c(1, 1), mar = c(5, 5, 4, 2))
  plot_idf_curve(results, return_periods = c(2, 5, 10, 25, 50, 100))

  # Page 3: Time series for selected durations
  if (!is.null(annual_max_df)) {
    par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))
    for (dur in c("5min", "1hr", "6hr", "24hr")) {
      if (dur %in% names(annual_max_df)) {
        plot_annual_maxima_ts(annual_max_df, dur)
      }
    }
  }

  dev.off()

  if (verbose) {
    cat(sprintf("\n✓ Plots exported to: %s\n", output_path))
  }
}

#' Create comparison table for output
#' @param results Data frame from analyze_all_durations
#' @param annual_max_df Data frame from calculate_annual_maxima
#' @param return_period Return period for table
#' @return Data frame suitable for display
#' @export
create_summary_table <- function(results, annual_max_df, return_period = 100) {

  rl_col <- paste0("RL", return_period)

  # Calculate observed max for each duration
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
