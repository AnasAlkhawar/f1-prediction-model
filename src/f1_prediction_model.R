# Load the packages
library(dplyr)
library(ggplot2)
library(readr)
library(tidyr)
library(lubridate) 

# Importing datasets
drivers <- read_csv("drivers.csv")
constructorResults <- read_csv("constructorResults.csv")
constructorStandings <- read_csv("constructorStandings.csv")
constructors <- read_csv("constructors.csv")
driverStandings <- read_csv("driverStandings.csv")
lapTimes <- read_csv("lapTimes.csv")
pitStops <- read_csv("pitStops.csv")
qualifying <- read_csv("qualifying.csv")
races <- read_csv("races.csv")
results <- read_csv("results.csv")
seasons <- read_csv("seasons.csv")
status <- read_csv("status.csv")
circuits <- read_csv("circuits.csv")

# Adjusting the code to use dmy() for date parsing
drivers <- drivers %>%
  mutate(dob_parsed = dmy(dob, quiet = TRUE), # Using dmy() for day/month/year format
         age = ifelse(!is.na(dob_parsed), as.integer(interval(dob_parsed, today()) / years(1)), NA))

# Cleaning lapTimes dataset
lapTimes <- lapTimes %>%
  filter(!is.na(time))  # Remove rows where lap time is missing


# Cleaning pitStops dataset
pitStops <- pitStops %>%
  filter(!is.na(duration))  # Remove rows where pit stop duration is missing

# Cleaning qualifying dataset
qualifying <- qualifying %>%
  filter(!is.na(position))  # Remove rows where qualifying position is missing


# Cleaning races dataset
races <- races %>%
  filter(!is.na(year) & !is.na(name))  # Ensure each race has a year and a name


# Cleaning results dataset
results <- results %>%
  filter(!is.na(positionOrder))  # Remove rows where race position is missing

# Cleaning seasons dataset
seasons <- seasons %>%
  filter(!is.na(year))  # Ensure each season has a year


# Cleaning status dataset
status <- status %>%
  filter(!is.na(statusId) & !is.na(status))  # Ensure each status entry is complete

# Cleaning constructors dataset
constructors <- constructors %>%
  filter(!is.na(name))  # Ensure each constructor has a name

# Cleaning the constructorResults dataset
constructorResults <- constructorResults %>%
  filter(!is.na(raceId) & !is.na(constructorId))  # Ensure essential IDs are not missing

# Additional step if you need to handle 'NULL' values in the 'status' column
constructorResults <- constructorResults %>%
  mutate(status = ifelse(status == "NULL" | status == "", NA, status))  # Replace 'NULL' or empty with NA


# Cleaning the constructorStandings dataset
constructorStandings <- constructorStandings %>%
  filter(!is.na(constructorStandingsId) & !is.na(raceId) & !is.na(constructorId))  # Ensure key IDs are not missing

# If the unnamed eighth column is not needed, can remove it
constructorStandings <- select(constructorStandings, -`...8`)


#joining drivers and driverStandings
driver_standings_detailed <- driverStandings %>%
  inner_join(drivers, by = "driverId")

#constructorStandings and constructors
constructor_standings_detailed <- constructorStandings %>%
  inner_join(constructors, by = "constructorId")

#results and races
race_results_detailed <- results %>%
  inner_join(races, by = "raceId")

#driver_standings_detailed and race_results_detailed
driver_race_performance <- driver_standings_detailed %>%
  inner_join(race_results_detailed, by = c("raceId", "driverId"))


#lapTimes with races and drivers
lap_times_detailed <- lapTimes %>%
  inner_join(races, by = "raceId") %>%
  inner_join(drivers, by = "driverId")

#pitStops with races and drivers
pit_stops_detailed <- pitStops %>%
  inner_join(races, by = "raceId") %>%
  inner_join(drivers, by = "driverId")

summary(drivers)
summary(lapTimes)
summary(results)


ggplot(drivers, aes(x = age)) +
  geom_histogram(binwidth = 1, fill = "blue", color = "black") +
  labs(title = "Distribution of Driver Ages", x = "Age", y = "Count")

avg_lap_times <- lapTimes %>%
  group_by(raceId) %>%
  summarize(avgTime = mean(time))

ggplot(avg_lap_times, aes(x = raceId, y = avgTime)) +
  geom_line() +
  labs(title = "Average Lap Times per Race", x = "Race ID", y = "Average Time")

driver_wins <- results %>%
  filter(positionOrder == 1) %>%
  group_by(driverId) %>%
  summarize(totalWins = n())

ggplot(driver_wins, aes(x = driverId, y = totalWins)) +
  geom_bar(stat = "identity") +
  labs(title = "Total Wins per Driver", x = "Driver ID", y = "Total Wins")


driver_performance <- results %>%
  group_by(driverId) %>%
  summarize(totalWins = sum(positionOrder == 1, na.rm = TRUE),
            totalPodiums = sum(positionOrder %in% 1:3, na.rm = TRUE))

constructor_performance <- constructorStandings %>%
  group_by(constructorId) %>%
  summarize(totalPoints = sum(points, na.rm = TRUE),
            totalWins = sum(wins, na.rm = TRUE))

lap_time_trend <- lapTimes %>%
  inner_join(races, by = "raceId") %>%
  group_by(year) %>%
  summarize(avgLapTime = mean(time, na.rm = TRUE))

ggplot(lap_time_trend, aes(x = year, y = avgLapTime)) +
  geom_line() +
  labs(title = "Average Lap Times Over Years", x = "Year", y = "Average Lap Time")

pit_stop_analysis <- pitStops %>%
  group_by(raceId) %>%
  summarize(avgStops = mean(stop, na.rm = TRUE))

ggplot(pit_stop_analysis, aes(x = raceId, y = avgStops)) +
  geom_bar(stat = "identity") +
  labs(title = "Average Number of Pit Stops per Race", x = "Race ID", y = "Average Stops")


constructor_evolution <- races %>%
  inner_join(results, by = "raceId") %>%
  group_by(year) %>%
  summarize(numConstructors = n_distinct(constructorId))

ggplot(constructor_evolution, aes(x = year, y = numConstructors)) +
  geom_line() +
  labs(title = "Number of Constructors Over the Years", x = "Year", y = "Number of Constructors")

# Join drivers with results first
drivers_results <- drivers %>%
  inner_join(results, by = "driverId")

# Now join this with races
drivers_with_age <- drivers_results %>%
  inner_join(races, by = "raceId") %>%
  mutate(dob = as.Date(dob_parsed),  # Use 'dob_parsed' as it's already in Date format
         age_at_race = as.integer(as.Date(date) - dob) / 365.25)  # Calculate age at race

# View the structure of the new dataset
str(drivers_with_age)

# Select relevant columns for correlation analysis
relevant_data <- drivers_with_age %>%
  select(age_at_race, positionOrder, points, laps, grid)  # Specify actual column names



# Ensure all selected features are numerical
relevant_data <- relevant_data %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

# Compute correlation matrix
correlation_matrix <- cor(relevant_data, use = "complete.obs")

# View the correlation matrix
print(correlation_matrix)


# Merge car specifications with consideration of unique keys
race_data_with_specs <- drivers_with_age %>%
  inner_join(constructorResults, by = c("raceId", "constructorId"))

# Merge circuit characteristics with consideration of unique keys
final_race_data <- race_data_with_specs %>%
  inner_join(circuits, by = "circuitId")


str(final_race_data)

# Extended correlation analysis with selected columns
extended_data <- final_race_data %>%
  select(age_at_race, positionOrder, points.x, laps, grid, constructorId, circuitId, lat, lng) %>%
  mutate_if(is.character, as.factor) %>%
  mutate_if(is.factor, as.numeric)

# Compute extended correlation matrix
extended_correlation_matrix <- cor(extended_data, use = "complete.obs")

# View the extended correlation matrix
print(extended_correlation_matrix)

# Replace missing numeric values with 0
numeric_columns <- sapply(final_race_data, is.numeric)
final_race_data[, numeric_columns] <- lapply(final_race_data[, numeric_columns], function(x) { replace(x, is.na(x), 0) })

# Replace missing categorical values with "Unknown" or another placeholder
categorical_columns <- sapply(final_race_data, function(x) { is.character(x) | is.factor(x) })
final_race_data[, categorical_columns] <- lapply(final_race_data[, categorical_columns], function(x) { replace(x, is.na(x), "Unknown") })


# Split the data into training and testing sets
set.seed(1) # for reproducibility
training_indices <- sample(1:nrow(final_race_data), 0.8 * nrow(final_race_data))

train_data <- final_race_data[training_indices, ]
test_data <- final_race_data[-training_indices, ]

# Linear regression model
model <- lm(points.x ~ age_at_race + grid + constructorId + circuitId + laps, data = train_data)

# Predict on the test set
predictions <- predict(model, test_data)

# Calculate RMSE
rmse <- sqrt(mean((test_data$points.x - predictions)^2))
print(paste("RMSE: ", rmse))


# Convert 'duration' to numeric if it's not already
pitStops$duration <- as.numeric(as.character(pitStops$duration))

# Distribution of pit stop durations
ggplot(pitStops, aes(x = duration)) +
  geom_histogram(binwidth = 1, fill = "red", color = "black") +
  labs(title = "Distribution of Pit Stop Durations")



# Distribution of key variables
ggplot(results, aes(x = points)) + geom_histogram(binwidth = 1, fill = "blue", color = "black") + labs(title = "Distribution of Points")

# Distribution of lap times
ggplot(lapTimes, aes(x = milliseconds)) + geom_histogram(fill = "green", color = "black") + labs(title = "Distribution of Lap Times")

# Distribution of pit stop durations
ggplot(pitStops, aes(x = duration)) + geom_histogram(fill = "red", color = "black") + labs(title = "Distribution of Pit Stop Durations")

# Data Exploration
# Summary statistics for numeric features
summary_stats <- sapply(final_race_data, function(x) if(is.numeric(x)) summary(x) else NA)

# Box plots for numeric features to identify outliers
boxplot_data <- final_race_data %>%
  select_if(is.numeric) %>%
  gather(key = "variables", value = "values")

ggplot(boxplot_data, aes(x = variables, y = values)) +
  geom_boxplot() +
  theme(axis.text.x = element_text(angle = 90, vjust = 0.5, hjust=1))

# Pairwise relationships of features
pairs(final_race_data[, sapply(final_race_data, is.numeric)])

# Correlation heatmap
library(corrplot)
correlation_matrix <- cor(final_race_data[, sapply(final_race_data, is.numeric)])
corrplot(correlation_matrix, method = "color")

# Feature Selection
# Using stepwise regression to select features
stepwise_model <- stepAIC(model, direction = "both")
if (!require(MASS)) {
  install.packages("MASS")
}

# Load the MASS package
library(MASS)

# Now you can use the stepAIC function
stepwise_model <- stepAIC(model, direction = "both")

# Alternatively, using random forest for feature importance
library(randomForest)
set.seed(123)
rf_model <- randomForest(points.x ~ ., data = train_data, importance = TRUE)
varImpPlot(rf_model)



if (!require(MASS)) {
  install.packages("MASS")
}

# Load the MASS package
library(MASS)

# Fit the final model without constructorId
final_model <- lm(points.x ~ age_at_race + grid + circuitId + laps, data = train_data)

# Summary of the final model
summary(final_model)

# Make predictions on the test data set
final_predictions <- predict(final_model, test_data)

# Evaluate the model's performance using RMSE
final_rmse <- sqrt(mean((test_data$points.x - final_predictions)^2))
print(paste("Final RMSE: ", final_rmse))


# Load the necessary library
library(randomForest)


# Fit the Random Forest model
rf_model <- randomForest(points.x ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)

# Print the model summary
print(rf_model)

# Predict on the test set
rf_predictions <- predict(rf_model, test_data)

# Calculate RMSE for the Random Forest model
rf_rmse <- sqrt(mean((test_data$points.x - rf_predictions)^2))
print(paste("Random Forest RMSE: ", rf_rmse))

importance(rf_model)
varImpPlot(rf_model)


# Impute missing values with the mean for numerical columns
train_data <- train_data %>%
  mutate_if(is.numeric, ~ifelse(is.na(.), mean(., na.rm = TRUE), .))

# Impute missing values with the mode for categorical columns
getmode <- function(v) {
  uniqv <- unique(v)
  uniqv[which.max(tabulate(match(v, uniqv)))]
}

train_data <- train_data %>%
  mutate_if(is.factor, ~ifelse(is.na(.), getmode(.), .))

# Now try fitting 
rf_model <- randomForest(points.x ~ ., data = train_data, ntree = 500, mtry = 3, importance = TRUE)

rf_model



# Impute or remove missing values in test data as done for training data
test_data <- na.omit(test_data)

# Predict on the test data
predictions <- predict(rf_model, test_data)

# Calculate the Root Mean Squared Error (RMSE) on test data
actual <- test_data$points.x
rmse <- sqrt(mean((actual - predictions)^2))
print(paste("Test RMSE: ", rmse))

# You can also calculate the R-squared on the test data to see how well the model performs
rsq <- cor(actual, predictions)^2
print(paste("Test R-squared: ", rsq))


# Let's say you have these values
linear_regression_rmse <- 3.20
linear_regression_rsq <- 0.57

random_forest_rmse <- 1.35
random_forest_rsq <- 0.97


# Create a data frame to hold the model comparison data
model_comparison <- data.frame(
  Model = c("Linear Regression", "Random Forest"),
  RMSE = c(linear_regression_rmse, random_forest_rmse),
  RSquared = c(linear_regression_rsq, random_forest_rsq)
)


# Print the comparison table
print(model_comparison)

# Plot RMSE for all models for visual comparison
ggplot(model_comparison, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison by RMSE", x = "Model", y = "RMSE")

# Plot R-squared for all models for visual comparison
ggplot(model_comparison, aes(x = Model, y = RSquared, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  labs(title = "Model Comparison by R-squared", x = "Model", y = "R-squared")

# Trying different values for mtry 
num_features <- ncol(train_data) - 1  # excluding the response variable
mtry_val <- sqrt(num_features)

rf_model_adjust_mtry <- randomForest(points.x ~ ., data = train_data, ntree = 500, mtry = mtry_val, importance = TRUE)


# Example for testing the model with more trees
test_predictions <- predict(rf_model_adjust_mtry, test_data)
test_rmse <- sqrt(mean((test_data$points.x - test_predictions)^2))
print(paste("Test RMSE with more trees: ", test_rmse))


# Predict on the Test Data
test_predictions <- predict(rf_model_adjust_mtry, test_data)

# Calculate the RMSE on Test Data
# Make sure 'points.x' is your target variable in the test data
test_rmse <- sqrt(mean((test_data$points.x - test_predictions)^2))
print(paste("Test RMSE: ", test_rmse))

# Print the OOB RMSE for comparison
oob_rmse <- sqrt(tail(rf_model$mse, 1))
print(paste("OOB RMSE: ", oob_rmse))

# Compare the RMSEs
if(test_rmse > oob_rmse) {
  print("Model might be overfitting as Test RMSE is higher than OOB RMSE.")
} else {
  print("Model seems to generalize well as Test RMSE is close to or lower than OOB RMSE.")
}

