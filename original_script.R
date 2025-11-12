################################################################################
## ORIGINAL SCRIPT - FOR REFERENCE
## This is the original monolithic script before modularization
## Kept for comparison and reference purposes
################################################################################

library(extRemes)

# Parametreler
DURATIONS <- c(5, 10, 15, 30, 60, 180, 360, 720, 1440)
DURATION_NAMES <- c("5min", "10min", "15min", "30min", "1hr", "3hr", "6hr", "12hr", "24hr")
RETURN_PERIODS <- c(2, 5, 10, 25, 50, 100)

# Yillik maksimumlar
yearly_data<-array(NA,c(360*288*86,4))

total_days=1

for(year in 1:86){
  for(month in 1:12){

    data_write_filled<-array(0,c(30*86,288))

    dis_data <- as.matrix(read.table(paste0("C:/Users/ser_o/OneDrive/Documents/EXTREMES/hyetos/hyetos/hyetosout_ssp245_ 31 _ ",month," _ 31 .txt")))[, 5:292]
    data_write_filled[(1):(dim(dis_data)[1]),]<-dis_data


    data_write<-data_write_filled[(year*30-29):(year*30),]


    for(day in 1:30){


      yearly_data[(total_days*288-287):(total_days*288),4]<-data_write[day,]

      total_days=total_days+1




    }

    print(c(year,month))
  }
}
yearly_data[is.na(yearly_data)]<-0
summary(yearly_data)
yearly_data[,3]<-1:288

years_write<-NA

for(year in 2015:2100){
  years_write[((year-2014)*103680-103679):((year-2014)*103680)]<-year

}

yearly_data[,2]<-years_write

years <- 2015:2100
calendar_vec <- c(as.Date("2014-01-04"))

for (yy in years) {
  # 360-day year starting on Jan 1 of that year
  days_360 <- as.Date(paste0(yy, "-01-01")) + 0:359

  # repeat each day 288 times
  days_360_rep <- rep(days_360, each = 288)

  # add to main vector
  calendar_vec <- c(calendar_vec, days_360_rep)
}

calendar_vec<-calendar_vec[-1]

yearly_data<-data.frame(yearly_data)

yearly_data[,1]<-(calendar_vec)

colnames(yearly_data)<-c("date","year","interval","precip")


##########################


# 2. Yillik maksimumlar
calculate_annual_max_wide <- function(df_long) {

  cat("\n=== YILLIK MAKSIMUMLAR ===\n")

  years <- sort(unique(df_long$year))
  n_years <- length(years)

  annual_max <- matrix(NA, nrow = n_years, ncol = length(DURATIONS))
  colnames(annual_max) <- DURATION_NAMES

  cat(sprintf("Hesaplaniyor: %d yil, %d duration...\n", n_years, length(DURATIONS)))

  for(yr_idx in 1:n_years) {

    yr <- years[yr_idx]

    # Yilin verisi
    year_data <- df_long$precip[df_long$year == yr]

    cat(sprintf("\n  Yil %d (%d/%d):\n", yr, yr_idx, n_years))
    cat(sprintf("    Toplam kayit: %d\n", length(year_data)))
    cat(sprintf("    Toplam yagis: %.2f mm\n", sum(year_data, na.rm = TRUE)))
    cat(sprintf("    Max 5-dk: %.2f mm\n", max(year_data, na.rm = TRUE)))

    for(d in 1:length(DURATIONS)) {

      dur_min <- DURATIONS[d]
      window <- dur_min / 5  # 5-dakikalik birim sayisi

      if(window == 1) {
        # 5 dakika - direkt maksimum
        annual_max[yr_idx, d] <- max(year_data, na.rm = TRUE)

      } else {
        # Moving window - yil boyunca surekli tara
        n <- length(year_data)

        if(n < window) {
          annual_max[yr_idx, d] <- NA
          cat(sprintf("    %s: Yetersiz veri\n", DURATION_NAMES[d]))
          next
        }

        # Tum ardisik toplamlar
        moving_sums <- sapply(1:(n - window + 1), function(i) {
          sum(year_data[i:(i + window - 1)], na.rm = TRUE)
        })

        annual_max[yr_idx, d] <- max(moving_sums, na.rm = TRUE)

        # Debug bilgisi (1hr ve 24hr icin)
        if(d %in% c(5, 9)) {
          cat(sprintf("    %s (window=%d): %.2f mm\n",
                      DURATION_NAMES[d], window, annual_max[yr_idx, d]))
        }
      }
    }
  }

  result <- data.frame(year = years, annual_max)

  cat("\n\n=== YILLIK MAKSIMUMLAR TABLOSU ===\n")
  print(result)

  # MANTIK KONTROLU
  cat("\n=== MANTIK KONTROLU ===\n")
  cat("Butun yillar icin maksimumlar:\n")
  for(d in 1:length(DURATIONS)) {
    max_val <- max(result[, d+1], na.rm = TRUE)
    cat(sprintf("  %-6s: %6.2f mm\n", DURATION_NAMES[d], max_val))
  }

  cat("\nMantik kontrol:\n")
  cat(sprintf("  5min < 1hr? %s (%.2f < %.2f)\n",
              ifelse(max(result[[2]]) < max(result[[6]]), "OK", "HATA"),
              max(result[[2]]), max(result[[6]])))
  cat(sprintf("  1hr < 24hr? %s (%.2f < %.2f)\n",
              ifelse(max(result[[6]]) < max(result[[10]]), "OK", "HATA"),
              max(result[[6]]), max(result[[10]])))

  return(result)
}



annual_max <- calculate_annual_max_wide(yearly_data)

# Sonuc tablosu
results <- data.frame(
  Duration = DURATION_NAMES,
  Mean = NA,
  SD = NA,
  Location = NA,
  Scale = NA,
  Shape = NA,
  RL2 = NA,
  RL5 = NA,
  RL10 = NA,
  RL25 = NA,
  RL50 = NA,
  RL100 = NA
)

# Her duration icin GEV fit et
for(i in 1:length(DURATION_NAMES)) {

  dur_name <- DURATION_NAMES[i]
  data_dur <- annual_max[, i + 1]


  # Gumbel fit (GEV with shape=0)

  fit <- tryCatch({
    fevd(data_dur, type = "Gumbel", method = "MLE")
  }, error = function(e) {
    cat("HATA:", e$message, "\n")
    return(NULL)
  })

  if(is.null(fit)) {
    cat("Model fit BASARISIZ!\n")
    next
  }

  # Parametreler
  params <- findpars(fit)

  # Return levels
  rl <- return.level(fit, return.period = RETURN_PERIODS)

  for(j in 1:length(RETURN_PERIODS)) {
    cat(sprintf("  RL-%3d: %6.2f mm\n", RETURN_PERIODS[j], rl[j]))
  }

  # Tabloya kaydet
  results$Mean[i] <- mean(data_dur)
  results$SD[i] <- sd(data_dur)
  results$Location[i] <- params$location
  results$Scale[i] <- params$scale
  results$Shape[i] <- 0.0  # Gumbel: shape = 0
  results$RL2[i] <- rl[1]
  results$RL5[i] <- rl[2]
  results$RL10[i] <- rl[3]
  results$RL25[i] <- rl[4]
  results$RL50[i] <- rl[5]
  results$RL100[i] <- rl[6]
}

# Ozet tablo

print(results[, c("Duration", "Location", "Scale", "Shape")])

rl100_table <- data.frame(
  Duration = results$Duration,
  Mean = round(results$Mean, 1),
  Max = round(c(
    max(annual_max$X5min),
    max(annual_max$X10min),
    max(annual_max$X15min),
    max(annual_max$X30min),
    max(annual_max$X1hr),
    max(annual_max$X3hr),
    max(annual_max$X6hr),
    max(annual_max$X12hr),
    max(annual_max$X24hr)
  ), 1),
  RL100 = round(results$RL100, 1)
)
print(rl100_table)

# MANTIK KONTROLU
for(i in 1:(nrow(results)-1)) {
  if(!is.na(results$RL100[i]) && !is.na(results$RL100[i+1])) {
    if(results$RL100[i] > results$RL100[i+1]) {
      cat(sprintf("WARNING: %s (%.1f) > %s (%.1f)\n",
                  results$Duration[i], results$RL100[i],
                  results$Duration[i+1], results$RL100[i+1]))
    } else {
      cat(sprintf("OK: %s (%.1f) < %s (%.1f)\n",
                  results$Duration[i], results$RL100[i],
                  results$Duration[i+1], results$RL100[i+1]))
    }
  }
}

# Grafik

par(mfrow = c(2, 2), mar = c(4, 4, 3, 2))

# 1. RL100 karsilastirma
barplot(results$RL100,
        names.arg = results$Duration,
        col = "steelblue",
        main = "100-Year Return Levels (Gumbel)",
        ylab = "Precipitation (mm)",
        las = 2)

# 2. Scale parameters
barplot(results$Scale,
        names.arg = results$Duration,
        col = "lightblue",
        main = "Scale Parameters (sigma)",
        ylab = "sigma",
        las = 2)

# 3. Mean vs RL100
plot(results$Mean, results$RL100,
     pch = 19, col = "darkblue", cex = 1.5,
     xlab = "Mean (mm)", ylab = "RL100 (mm)",
     main = "Mean vs RL100")
text(results$Mean, results$RL100, results$Duration, pos = 3, cex = 0.7)
abline(0, 1, col = "red", lty = 2)

# 4. IDF curve
plot(DURATIONS, results$RL100,
     type = "b", pch = 19, col = "darkgreen", lwd = 2,
     log = "xy",
     xlab = "Duration (min)", ylab = "RL100 (mm)",
     main = "IDF Curve - Gumbel (100-year)")
grid()

# CSV kaydet
write.csv(results, "stationary_gumbel_results.csv", row.names = FALSE)
