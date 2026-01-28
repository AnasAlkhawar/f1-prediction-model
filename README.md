# 🏎️ Formula 1 Race Performance Prediction Model (R) 📌 Overview

This project is an end-to-end Formula 1 race data analysis and prediction pipeline built in R, leveraging large-scale historical race, driver, constructor, lap time, pit stop, and circuit datasets to model and predict race performance (points scored).

The pipeline combines data cleaning, feature engineering, exploratory data analysis (EDA), and machine learning to compare traditional statistical models with ensemble methods, emphasizing reproducibility and real-world sports analytics.

---

## 🧠 Key Objectives

- Clean and integrate large, multi-table Formula 1 datasets  
- Analyze trends in driver age, lap times, pit stops, and constructor performance  
- Engineer features relevant to race outcomes  
- Build and compare Linear Regression and Random Forest models  
- Evaluate model performance using RMSE and R²  

---

## 📊 Datasets Used

The project uses historical Formula 1 datasets including:

- **Drivers & Constructors** (biographical and team data)  
- **Races & Circuits** (locations, years, track characteristics)  
- **Lap Times & Pit Stops** (race-level performance data)  
- **Results & Standings** (finishing positions, points, wins)  

All datasets are joined using consistent relational keys:
`driverId`, `constructorId`, `raceId`, `circuitId`.

---

## 🧹 Data Cleaning & Feature Engineering

Key preprocessing steps include:

- Parsing driver birthdates and computing **age at race**
- Removing incomplete or invalid lap times, pit stops, and race results
- Handling missing values using:
  - Mean imputation (numerical features)
  - Mode or placeholder imputation (categorical features)
- Converting categorical identifiers into numeric representations for modeling

### Engineered Features

- Grid position  
- Laps completed  
- Constructor and circuit identifiers  
- Geographic circuit attributes (latitude & longitude)  

---

## 📈 Exploratory Data Analysis (EDA)

The project includes extensive visualization and exploratory analysis, such as:

- Distribution of driver ages  
- Average lap time trends across seasons  
- Pit stop duration distributions  
- Driver wins and podium frequency  
- Constructor evolution over time  
- Correlation heatmaps for numerical features  
- Boxplots and pairwise feature relationships  

### Visualization Tools Used
- `ggplot2`
- `corrplot`

---

## 🤖 Modeling Approach

### 1️⃣ Linear Regression (Baseline Model)

A baseline regression model predicts race points using:

- Driver age at race  
- Grid position  
- Circuit ID  
- Laps completed  

**Evaluation Metrics**
- RMSE  
- R²  

---

### 2️⃣ Random Forest (Ensemble Model)

A more advanced Random Forest model is trained using all engineered features to capture non-linear relationships and feature interactions.

**Key Components**
- Hyperparameter tuning (`ntree`, `mtry`)
- Feature importance analysis
- Out-of-bag (OOB) error comparison
- Overfitting detection via Test vs OOB RMSE

---

## 📊 Model Performance Summary

| Model              | RMSE | R²  |
|-------------------|------|-----|
| Linear Regression | 3.20 | 0.57 |
| Random Forest     | 1.35 | 0.97 |

✅ **Random Forest significantly outperformed Linear Regression**, capturing complex performance dynamics in Formula 1 races.

---

## 🛠️ Technologies & Tools

- **Language:** R  
- **Data Manipulation:** `dplyr`, `tidyr`, `lubridate`, `readr`  
- **Visualization:** `ggplot2`, `corrplot`  
- **Modeling:** `lm`, `randomForest`, `MASS (stepAIC)`  
- **Environment:** RStudio, GitHub  

---

## 🚀 Future Improvements

- Add time-aware modeling (season-based train/test splits)
- Introduce weather and track condition features
- Replace numeric IDs with learned embeddings
- Build a Shiny dashboard for interactive predictions
- Port the pipeline to Python (scikit-learn / XGBoost)

---

## 👤 Author

**Anas Alkhawar**  
Computer Science & Data Science Graduate  
Focused on data engineering, machine learning, and real-world analytics systems

---

## 📎 Why This Project Matters

This repository demonstrates:

- End-to-end data science workflow  
- Strong data engineering fundamentals  
- Practical machine learning model evaluation  
- Ability to work with large, relational datasets  
- Clear documentation and reproducibility