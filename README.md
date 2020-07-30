# LTA Mobility Sensing Project for Land Transport Master Plan (LTMP) 2040
## Project Objective
Location information about commuter activities is vital for planning for travel disruptions and infrastructural development. The Mobility Sensing Project aims to find innovative and novel ways to identify travel patterns from GPS data and other multi-sensory data collected in smartphones. This will be transformative to provide personalised travel information. 
## Project Outline
### Part 1
- Retrieve data from AWS S3 bucket
- Download to local machine as csv file
### Part 2
- Load and merge 3 files into Python
- Perform initial cleaning phase such as removing duplicates and converting epoch to date and time
- Selected raw features: Accelerometer, Gyroscope, Magnetometer
### Part 3
- Statistical Feature-Based Approach
   * Data Exploration
      * Boxplots 
      * Density Plots
      * Pair plot
      * Aggregation Plots
          * Mean Plots
          * Median Plots
      * Time Series Plots
   * Data Cleaning/Denoising
      * Kalman Filter
   * Data Preparation/Scaling
      * Standardization (Z-score Normalization)
          * Standardize the training set using the training set means and standard deviations to prevent data leakage
              * Subtract mean, divide standard deviation
              * Necessary to normalize data before performing PCA
          * General principle: any thing you learn, must be learned from the model's training data
   * Feature Extraction/Selection
      * Main idea: Transformation of patterns into features that are considered as a compressed representation
      * For each time series variable, key statistical features will be extracted to measure different properties of that variable
      * Main groups of the calculated statistical measures (Non-exhaustive):
          * Measures of Central Tendency (Mean, Median)
          * Measures of Variability (Variance, Standard Deviation, IQR, Range)
          * Measures of Shape (Skewness, Kurtosis)
          * Measures of Position (Percentiles)
          * Measures of Impurity (Entropy)
              * Not ideal to be using the entropy function from sklearn as it assumes a discrete distribution of the data. Instead, we will be using a custom ready-made non-parametric k-nearest neighbour entropy estimator.
      * Determine a moving window size to calculate the above features (An initial size of window 10 is chosen based on literature)
      * Before extracting the features, a magnitude vector is calculated from each sensor to obtain 3 extra data sources
          * The formula for the magnitude of a vector can be generalized to arbitrary dimensions. For example, if a = (a1,a2,a3,a4) is a four-dimensional vector, the formula for its magnitude is ∥a∥ = √a21+a22+a23+a24.
   * Classifier Training/Tuning/Evaluation (Before and after upsampling/downsampling techniques)
      * Train-Test Split
          * Perform a 70/30 train test split with stratification to ensure balanced Y distribution in both train and test sets
          * Set seed for reproducible results
          * To minimize data leakage when developing predictive models:
              * Perform data preparation within training set
              * Hold back a test set for final sanity check of the developed models
       * Trial Modeling Stage
          * Before proceeding with PCA, build a few common classifiers using sklearn first. This allows us to compare the results before and after using PCA.
          * Perform cross validation on the training set to use less training time
          * Results might not be as good with less data but that doesn't matter for now as it is more about gaining insights into as many models as possible first.
          * Keep the default settings for most models
          * Models tested: LogisticRegression, SVC, LinearSVC, KNeighborsClassifier, DecisionTreeClassifier, RandomForestClassifier, GaussianNB, MLPClassifier
          * Random Forest performed the best (94.866%) in terms of mean F1 Score with a 5 fold cross validation, followed by Support Vector Machine (SVM) with rbf kernel and MLPClassifier.
          * Insights:
              * Due to the class imbalance, accuracy is not a good metric to be used here. F1 Score is a more suitable performance metric. 
              * Upsampling techniques may be required too later to solve the class imbalance issue.
              * Summary of trial modeling results
                  * Before PCA (216 features)
              
                    |**Model**          |**F1 Score**|**Standard Deviation (%)**|
                    |-------------------|------------|--------------------------|
                    | LogisticRegression| 0.91717    | 0.48                     |
                    | SVC               | 0.94076    | 1.64                     |
                    | LinearSVC         | 0.91584    | 0.89                     |
                    | KNeighbors        | 0.92328    | 1.62                     |
                    | DecisionTree      | 0.92687    | 2.91                     |
                    | RandomForest      | 0.94866    | 3.25                     |
                    | GaussianNB        | 0.33205    | 14.72                    |
                    | MLPClassifier     | 0.93195    | 1.62                     |
                
                  * After PCA (48 features)
                  
                    |**Model**          |**F1 Score**|**Standard Deviation (%)**|
                    |-------------------|------------|--------------------------|
                    | LogisticRegression| 0.92014    | 0.72                     |
                    | SVC               | 0.93772    | 1.43                     |
                    | LinearSVC         | 0.91704    | 0.25                     |
                    | KNeighbors        | 0.92413    | 1.58                     |
                    | DecisionTree      | 0.93004    | 3.22                     |
                    | RandomForest      | 0.95004    | 3.20                     |
                    | GaussianNB        | 0.52965    | 20.92                    |
                    | MLPClassifier     | 0.93880    | 2.56                     |
                  
          * Boosting Algorithms tried
              * AdaBoost (Adaptive Boosting)
              * CatBoost
                  * Hyper-parameter tuning is seldom needed when using CatBoost.
                  * CatBoost which is implemented by powerful theories like ordered Boosting, Random permutations, makes sure that we are not overfitting our model. It also implements symmetric trees which eliminates parameters like (min_child_leafs). We can further tune with parameters like learning_rate, random_strength, L2_regulariser, but the results usually doesn’t vary much.
              * Light GBM
                  * Gradient boosting framework that uses tree based learning algorithms
                  * Designed to be distributed and efficient with the following advantages:
                      * Faster training speed and higher efficiency
                      * Lower memory usage
                      * Better accuracy
                      * Support of parallel and GPU learning
                      * Capable of handling large-scale data
               * Insights:
                  * F1 macro-average will be used in place of accuracy or F1-weighted based on different considerations for precision-recall tradeoff
                  * F1 macro-average will compute the metric independently for each class and then take the average (hence treating all classes equally). This is ideal for imbalanced datasets.
                  * As Light GBM only takes around 10-15 seconds to run and the default parameters already look quite promising, it is worth tuning the hyperparameters to aim for better results. We will attempt random search first, which is more flexible and more efficient than a grid search. We will also utilize the trainset and testset now.
                  * If the close-to-optimal region of hyperparameters occupies at least 5% of the grid surface, then random search with 60 trials will find that region with high probability (95%). Higher number can be used if running time is low enough to avoid unlucky searches.
                  * Best Average F1 Macro reached: 0.7271336837697994
                      * The lower score compared to normal F1 Score used by earlier models indicate that it is more ideal for tuning as it recognized the precision-recall tradeoff and returned a lower score. Hopefully, the hyperparameters are tuned to balance precision and recall better.
                  * Optimal parameters for the best estimator
                      * {'boosting_type': 'gbdt', 'class_weight': None, 'colsample_bytree': 0.5989147519171187, 'importance_type': 'split', 'is_unbalance': True,
'learning_rate': 0.1, 'max_depth': -1, 'min_child_samples': 152, 'min_child_weight': 0.01, 'min_split_gain': 0.0,
'n_estimators': 150, 'n_jobs': -1, 'num_leaves': 49, 'objective': 'binary', 'random_state': 0,
'reg_alpha': 2, 'reg_lambda': 0, 'silent': True, 'subsample': 0.8711234237005314, 'subsample_for_bin': 200000, 'subsample_freq': 0}
                  
                  * Confusion Matrix Definition
                
                    |**Error**            |**Definition**                      |
                    |---------------------|------------------------------------|
                    | True Positives (TP) | Predicted Not MRT correctly        |
                    | True Negatives (TN) | Predicted MRT correctly            |
                    | False Positives (FP)| Predicted Not MRT but actual is MRT|
                    | False Negatives (FN)| Predicted MRT but actual is Not MRT|
                    
                  * Confusion Matrix Results
                   
                    |**Error**            |**Values**|
                    |---------------------|----------|
                    | True Positives (TP) | 18986    |
                    | True Negatives (TN) | 752      |
                    | False Positives (FP)| 377      |
                    | False Negatives (FN)| 115      |
                    
                  * The model optimizes recall instead of precision. In this case, recall can be thought as of a model’s ability to find all the data points of interest (MRT) in a dataset. A precision-recall tradeoff is common in many scenarios and it often boils down to the business problem that the company wants to solve or improve on.
                   
                  * Final LightGBM with random search
                   
                    |**Metrics**|**Score**|
                    |-----------|---------|
                    |Accuracy   | 0.9778  |
                    |F1 Score   | 0.9883  |
                    |Precision  | 0.9812  |
                    |Recall     | 0.9956  |
                   
## Code and Resources Used
- **Database:** AWS S3
- **Packages:** json, io, boto3, pandas, numpy, matplotlib, seaborn, datetime, sklearn, math, scipy, catboost, lightgbm, imblearn, hyperopt, xgboost, vecstack 
- Special thanks to **Guanlan** for Kalman Filter script