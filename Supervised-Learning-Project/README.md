## Hotel Booking Cancellation Prediction

This project aims to predict hotel booking cancellations using a supervised machine learning pipeline. The notebook details the process from data loading and cleaning to model training, evaluation, and hyperparameter tuning to identify the best-performing model for this classification task.

## Project Workflow

The project follows these key steps:

1.  **Environment Setup and Data Loading**
    *   **Libraries Imported**: `pandas` for data manipulation, `numpy` for numerical operations, `matplotlib` and `seaborn` for visualization, and `sklearn` for machine learning tasks.
    *   **Data Loaded**: The dataset `data_hotel_booking_demand.csv` was loaded into a pandas DataFrame.

2.  **Exploratory Data Analysis and Preprocessing**
    *   **Initial Inspection**:
        *   Displayed the first few rows (`df.head()`), dimensions (`df.shape`), data types, and non-null counts (`df.info()`).
        *   Generated descriptive statistics for numerical features (`df.describe().T`).
    *   **Missing Value Handling**:
        *   Identified columns with missing values (`df.isnull().sum()`). 'AgentID' and 'CompanyID' had a high percentage of missing values and were dropped.
        *   'Country of Origin' missing values were imputed using the mode.
        *   'Children' missing values were imputed using the mean and the column was converted to an integer type.
    *   **Data Cleaning**:
        *   Rows where the sum of 'Adults', 'Children', and 'Babies' was zero (indicating no guests) were removed, as these are likely erroneous entries.
        *   Duplicate rows were identified and removed (`df.drop_duplicates()`).
    *   **Feature Removal/Transformation**:
        *   The 'ArrivalYear' column was dropped as all bookings were found to be for the year 2016, offering no predictive value.
        *   Identifier columns ('BookingID'), high cardinality/ PII columns ('Email', 'Phone', 'CreditCard') were dropped as they are not suitable for direct use in modeling or could lead to overfitting/privacy concerns.
    *   **Target Variable Analysis**:
        *   The distribution of the target variable 'Canceled' was examined, showing approximately 37.5% cancellations and 62.5% non-cancellations.

3.  **Feature Engineering**
    *   **Categorical Feature Encoding**:
        *   Categorical columns ('Country of Origin', 'MealPlan', 'RoomType', 'MarketSegment', 'BookingChannel') were identified.
        *   One-hot encoding (`pd.get_dummies()`) was applied to these categorical features, with `drop_first=True` to avoid multicollinearity. This transformed the categorical data into a numerical format suitable for machine learning algorithms.

4.  **Data Splitting and Scaling**
    *   **Train-Test Split**: The dataset was split into training (70%) and testing (30%) sets (`train_test_split` with `random_state=42` for reproducibility). `X` contained the features and `y` contained the target variable 'Canceled'.
    *   **Feature Scaling**: Numerical features in the training and testing sets were standardized using `StandardScaler`. This ensures that features with larger value ranges do not dominate those with smaller ranges during model training.

5.  **Model Training and Initial Evaluation**
    *   A custom function `model_fit_predict` was defined to streamline the training and evaluation process for multiple classifiers. This function fits the model, makes predictions, and prints/plots:
        *   Accuracy Score
        *   ROC AUC Score
        *   Classification Report (Precision, Recall, F1-Score)
        *   Confusion Matrix
    *   The following classification models were trained and evaluated:
        *   Logistic Regression (`solver='liblinear'`)
        *   K-Nearest Neighbors (KNN)
        *   Decision Tree (`random_state=42`)
        *   Random Forest (`random_state=42`)
        *   AdaBoost (`random_state=42`)
        *   Gradient Boosting (`random_state=42`)
        *   XGBoost (`random_state=42`, `eval_metric='logloss'`, `use_label_encoder=False`)
    *   *Note: A Support Vector Classifier (SVC) was initially included but commented out, likely due to its potentially long training time on this dataset.*

6.  **Model Performance Comparison**
    *   The ROC AUC scores and Accuracy scores for all trained models were compiled into DataFrames and sorted to identify the top-performing models.
    *   XGBoost, Gradient Boosting, and Random Forest generally showed strong initial performance.

7.  **Hyperparameter Tuning**
    *   GridSearchCV was employed to find the optimal hyperparameters for the top-performing models: XGBoost, Gradient Boosting, and Random Forest.
    *   The scoring metric used for tuning was 'roc_auc'.
    *   **XGBoost Tuning**: Parameters like `n_estimators`, `max_depth`, `learning_rate`, `subsample`, `colsample_bytree` were tuned.
    *   **Gradient Boosting Tuning**: Parameters like `n_estimators`, `max_depth`, `learning_rate`, `subsample` were tuned.
    *   **Random Forest Tuning**: Parameters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` were tuned.
    *   The best estimator from each `GridSearchCV` was then evaluated on the test set.

8.  **Final Model Selection and Feature Importance**
    *   The tuned models were compared based on their ROC AUC scores on the test set.
    *   The best-performing model after hyperparameter tuning was selected as the final model (the notebook output suggests the tuned XGBoost model performed best).
    *   Feature importances were extracted from the final model (e.g., `best_xgb.feature_importances_`) and visualized to understand which features contributed most to the predictions. Features such as 'BookingToArrivalDays', 'CountryofOriginHDI (Year-1)', 'DailyRateEuros', and various market segment encodings were among the important ones.

9.  **Conclusion**
    *   The notebook concludes by summarizing the performance of the best model (tuned XGBoost) and highlighting its effectiveness in predicting hotel booking cancellations. Key metrics (Accuracy and ROC AUC) for the final model were reported. The analysis of feature importances provides insights into the drivers of cancellations.

## How to Use

1.  Ensure all required libraries (pandas, numpy, sklearn, matplotlib, seaborn, xgboost) are installed.
2.  Place the `data_hotel_booking_demand.csv` file in the same directory as the notebook.
3.  Run the Jupyter notebook cells sequentially to reproduce the analysis and results.
    *   The hyperparameter tuning cells (especially for Gradient Boosting and Random Forest, and XGBoost with extensive grids) can be computationally intensive and may take some time to run.

