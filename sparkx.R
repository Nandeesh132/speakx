# Load necessary libraries
library(tidyverse)
library(caret)
library(ggplot2)

# Load the dataset
data <- read.csv(file.choose())
View(data)

# View the first few rows of the dataset
head(data)

# Check for missing values
colSums(is.na(data))

# Get the structure of the dataset
str(data)

# Remove rows with any missing values
data_cleaned <- na.omit(data)
str(data_cleaned)

# Distribution of target variable 'Churn'
ggplot(data_cleaned, aes(x = Churn, fill = Churn)) +
  geom_bar() +
  ggtitle('Distribution of Churn') +
  theme_minimal() +
  scale_fill_brewer(palette = "Set1")

# Distribution of categorical variables
categorical_vars <- data_cleaned %>% select_if(is.character)

for (var in colnames(categorical_vars)) {
  print(
    ggplot(data_cleaned, aes_string(x = var, fill = var)) +
      geom_bar() +
      ggtitle(paste('Distribution of', var)) +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")
  )
}

# Distribution of numerical variables
numerical_vars <- data_cleaned %>% select_if(is.numeric)

for (var in colnames(numerical_vars)) {
  print(
    ggplot(data_cleaned, aes_string(x = var)) +
      geom_histogram(bins = 30, fill = "skyblue", color = "black") +
      ggtitle(paste('Distribution of', var)) +
      theme_minimal()
  )
}

# Relationship between numerical variables and Churn
for (var in colnames(numerical_vars)) {
  print(
    ggplot(data_cleaned, aes_string(x = var, fill = 'Churn')) +
      geom_histogram(bins = 30, color = "black", position = "dodge") +
      ggtitle(paste('Relationship between', var, 'and Churn')) +
      theme_minimal() +
      scale_fill_brewer(palette = "Set1")
  )
}

# Convert categorical variables to factors
dataf <- data_cleaned %>%
  mutate_if(is.character, as.factor)

# Encode categorical variables using one-hot encoding
dataf <- data_cleaned %>%
  mutate_at(vars(-Churn), funs(as.numeric(as.factor(.))))

# Check the new structure of the dataset
str(dataf)

# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(dataf$Churn, p = .8, 
                                  list = FALSE, 
                                  times = 1)
train_data <- dataf[trainIndex, ]
test_data <- dataf[-trainIndex, ]

# Logistic Regression
logistic_model <- train(Churn ~ ., data = train_data, method = "glm", family = "binomial")
logistic_pred <- predict(logistic_model, test_data)
logistic_results <- confusionMatrix(logistic_pred, test_data$Churn)

# Random Forest
rf_model <- train(Churn ~ ., data = train_data, method = "rf")
rf_pred <- predict(rf_model, test_data)
rf_results <- confusionMatrix(rf_pred, test_data$Churn)

# Gradient Boosting
gbm_model <- train(Churn ~ ., data = train_data, method = "gbm", verbose = FALSE)
gbm_pred <- predict(gbm_model, test_data)
gbm_results <- confusionMatrix(gbm_pred, test_data$Churn)

# Print evaluation metrics
logistic_results
rf_results
gbm_results

# Visualize the ROC curves for the models
library(pROC)

# Logistic Regression ROC curve
logistic_roc <- roc(test_data$Churn, as.numeric(logistic_pred))
plot(logistic_roc, col = "blue", main = "ROC Curves for Churn Prediction Models")
legend("bottomright", legend = c("Logistic Regression"), col = c("blue"), lwd = 2)

# Random Forest ROC curve
rf_roc <- roc(test_data$Churn, as.numeric(rf_pred))
plot(rf_roc, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Random Forest"), col = c("blue", "red"), lwd = 2)

# Gradient Boosting ROC curve
gbm_roc <- roc(test_data$Churn, as.numeric(gbm_pred))
plot(gbm_roc, col = "green", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Random Forest", "Gradient Boosting"), col = c("blue", "red", "green"), lwd = 2)
