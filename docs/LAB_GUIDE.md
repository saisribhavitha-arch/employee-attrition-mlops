# Lab Guide — Employee Attrition Prediction

## Lab 1 — Data Loading
Read the employee dataset, inspect shape, columns, data types, and class distribution.

## Lab 2 — Data Validation
Check missing values, duplicates, invalid target labels, and basic numeric ranges.

## Lab 3 — Preprocessing
Separate target and features, split with stratification, impute numeric/categorical fields, scale numeric values, and one-hot encode categories.

## Lab 4 — Feature Engineering
Add tenure bands and income bands; save the engineered dataset.

## Lab 5 — Model Training
Train Logistic Regression and Random Forest pipelines; save the best model.

## Lab 6 — Evaluation
Generate accuracy, precision, recall, F1, ROC-AUC, confusion matrix, and classification report.

## Lab 7 — Experiment Tracking
Record model name, parameters, metrics, timestamp, and artifact path in a JSON experiment log.

## Lab 8 — Model Registry
Register a versioned model metadata record and mark the active version.

## Lab 9 — Deployment Preparation
Create a deployment manifest containing model path, feature names, target labels, and evaluation metrics.

## Lab 10 — API / Serving
Expose a FastAPI health endpoint and an employee attrition prediction endpoint.

## Lab 11 — Docker
Build a container image and run the API in a container.

## Lab 12 — Monitoring, Drift, and Retraining
Compare incoming batch feature distributions with the training reference, record drift indicators, evaluate the current model, train a candidate, and promote it when its F1 score is not lower.

## Lab 13 — End-to-End Automation
Run data checks, preprocessing, training, evaluation, experiment logging, registry update, and deployment preparation in one reproducible command.
