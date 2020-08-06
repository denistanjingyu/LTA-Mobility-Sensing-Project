# LTA Mobility Sensing Project for Land Transport Master Plan (LTMP) 2040

![ltmp-public-consultation-document-banner](https://user-images.githubusercontent.com/45563371/88942744-41a1dc00-d2bd-11ea-980c-04ffdf18728c.jpg)

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
      * Countplot
      
      ![image](https://user-images.githubusercontent.com/45563371/88942408-ce986580-d2bc-11ea-8c29-b346aa4e4de8.png)
      
      * Boxplots 
      
      ![image](https://user-images.githubusercontent.com/45563371/88942861-6433f500-d2bd-11ea-9d19-c0de9a460e49.png)
      
      * Density Plots
      
      ![image](https://user-images.githubusercontent.com/45563371/88942983-84fc4a80-d2bd-11ea-9914-0f47744e73b1.png)

      * Pair plot
      
      ![image](https://user-images.githubusercontent.com/45563371/88943109-a78e6380-d2bd-11ea-9d24-0ec2b9a51aa5.png)

      * Aggregation Plots
          * Mean Plots
          
          ![image](https://user-images.githubusercontent.com/45563371/88943194-c12fab00-d2bd-11ea-91f4-0e37f64951c4.png)

          * Median Plots
          
          ![image](https://user-images.githubusercontent.com/45563371/88943249-d573a800-d2bd-11ea-8986-5c473d2e32c9.png)

      * Time Series Plots
      
      ![image](https://user-images.githubusercontent.com/45563371/88943384-f89e5780-d2bd-11ea-94c4-4f1131c94ee8.png)

   * Data Denoising
      * Kalman Filter
          * Kalman filtering is an algorithm that provides estimates of some unknown variables given the measurements observed over time
          * Kalman filter algorithm consists of two stages: prediction and update
      
      ![image](https://user-images.githubusercontent.com/45563371/88941940-300c0480-d2bc-11ea-8d83-5f4718160e01.png)
      
   * Data Preparation/Scaling
      * Standardization (Z-score Normalization)
          * Standardize the training set using the training set means and standard deviations to prevent data leakage
              * Subtract mean, divide standard deviation
              * Necessary to normalize data before performing PCA
          * General principle: any thing you learn, must be learned from the model's training data
      * Apply Principal Component Analysis (PCA) on the scaled dataset
          * Instead of choosing the number of components manually, we will be using the option that allows us to set the variance of the input that is supposed to be explained by the generated components.
          * Typically, we want the explained variance to be between 95–99%. We will use 95% here.
          * As usual to prevent data leakage, we fit PCA on the training data set, and then we transform the test data set using the already fitted pca.
          * Cumulative Summation of the Explained Variance
          
          ![image](https://user-images.githubusercontent.com/45563371/88943791-80846180-d2be-11ea-9dff-0244cec7ee29.png)

          * Most important feature of each principal component
          
          ![image](https://user-images.githubusercontent.com/45563371/88943958-b590b400-d2be-11ea-8cd5-ac5f91b63536.png)

          * Number of times each sensor type appeared as the most important feature of a principal component
          
            |**Sensor Type**|**Count**|
            |---------------|---------|
            | Accelerometer | 15      |
            | Gyroscope     | 21      |
            | Magnetometer  | 12      |
            
          * Number of times each statistical type appeared as the most important feature of a principal component
          
            |**Statistical Type**|**Count**|
            |--------------------|---------|
            | Mean               | 5       |
            | Variance           | 8       |
            | Skewness           | 11      |
            | Kurtosis           | 13      |
            | Others             | 11      |
            
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
                    
      * Experimentation with oversampling and undersampling techniques
          * ADASYN: Adaptive Synthetic Sampling Method for Imbalanced Data
              * ADASYN works similarly to the regular SMOTE. However, the number of samples generated for each x_i is proportional to the number of samples which are not from the same class than x_i in a given neighborhood. Therefore, more samples will be generated in the area that the nearest neighbor rule is not respected.
           * Borderline-SMOTE: Over-Sampling Method in Imbalanced Data Sets
               * Only the minority examples near the borderline are over-sampled unliked the normal SMOTE and ADASYN.
           * Original vs Upsample (SMOTE) Scatterplot
               * Original data points
               
                 ![image](https://user-images.githubusercontent.com/45563371/88944787-bbd36000-d2bf-11ea-817a-d2b375c77f61.png)
           
               * Upsample (SMOTE) data points
               
                 ![image](https://user-images.githubusercontent.com/45563371/88944830-cee63000-d2bf-11ea-9f81-5bc1a1ec726d.png)
                 
           * Original vs Downsample Scatterplot
               * Original data points
               
                 ![image](https://user-images.githubusercontent.com/45563371/88945128-300e0380-d2c0-11ea-958e-8a3d060968e5.png)
           
               * Downsample data points
               
                 ![image](https://user-images.githubusercontent.com/45563371/88945179-3ef4b600-d2c0-11ea-94aa-9c1b7c383882.png)
           
      * LGBM using hyperopt with smote dataset
          * fmin - main function to minimize
          * tpe and anneal - optimization approaches
              * TPE (Tree-structured Parzen Estimator) is a default algorithm for the Hyperopt. It uses Bayesian approach for optimization. At every step it is trying to build probabilistic model of the function and choose the most promising parameters for the next step.
          * hp - include different distributions of variables
          * Trials - used for logging
          * Insights:
              * Results are slightly better than using random search
              * False positives are higher by an insignificant amount
      * Ensemble learning using heterogeneous algorithms
      
      ![ensemble-framework-packt](https://user-images.githubusercontent.com/45563371/89557512-82619e00-d845-11ea-81c0-4895fe38b692.jpg)

          * Ensemble methods use multiple learning algorithms to obtain better predictive performance than could be obtained from any of the constituent learning algorithms alone.
          * Test out the ensemble using different base algorithms.
          * Ensemble 1
              * Final LightGBM with SMOTE borderline and bayes optimisation (hyperopt)
              * Random Forest
          * Ensemble vote methods
              * In hard voting (also known as majority voting), every individual classifier votes for a class, and the majority wins. In statistical terms, the predicted target label of the ensemble is the mode of the distribution of individually predicted labels.
              * In soft voting, every individual classifier provides a probability value that a specific data point belongs to a particular target class. The predictions are weighted by the classifier's importance and summed up. Then the target label with the greatest sum of weighted probabilities wins the vote.
              * Comparison of F1 Macro results
              
                |**Model**              |**Score**|
                |-----------------------|---------|
                |lgbm_hyperopt_final    | 0.9816  |
                |Random Forest          | 0.9788  |
                |Voting_Classifier_Hard | 0.9812  |
                |Voting_Classifier_Soft | 0.9824  |
                    
              * The ensemble which used a soft voting system performed the best.
                  * False positive is quite high although overall result is good.
          * Ensemble 2 (Attempt to combine low correlation models to get higher precision score)
              * Final LightGBM with SMOTE borderline and bayes optimisation (hyperopt)
              * Random Forest
              * K-Nearest Neighbour
          * Only hard voting since k_nearest neighbour does not output any probabilities.
          * Insight:
              * Adding KNeighborsClassifier to the ensemble reduced false positives only by a little bit while false negatives close to doubled.
          * Creating a Stacking ensemble
          
            ![An-example-scheme-of-stacking-ensemble-learning](https://user-images.githubusercontent.com/45563371/89557165-fbacc100-d844-11ea-8449-1daf9ff07e8c.png)
          
              * The main idea behind the structure of a stacked generalization is to use one or more first level models, make predictions using these models and then use these predictions as features to fit one or more second level models on top. To avoid overfitting, cross-validation is usually used to predict the OOF (out-of-fold) part of the training set.
              * Train a normal xgboost first.
              * Define first level models. We will use light gbm, random forest and K-Nearest Neighbour.
              * Use first level models to make predictions.
              * Fit the second level model on new S_train and same Y_train_smote.
              * Results are around the same as just the lgbm model alone. Probably due to the other models not being highly complementary.
                
## Code and Resources Used
- **Database:** AWS S3
- **Packages:** json, io, boto3, pandas, numpy, matplotlib, seaborn, datetime, sklearn, math, scipy, catboost, lightgbm, imblearn, hyperopt, xgboost, vecstack 
- Special thanks to **Guanlan** for Kalman Filter script
