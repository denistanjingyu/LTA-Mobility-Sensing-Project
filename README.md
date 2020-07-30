# LTA Mobility Sensing Project
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
   * Data Cleaning/Denoising
   * Data Preparation/Scaling
   * Feature Extraction/Selection
   * Classifier Training/Tuning/Evaluation (Before and after upsampling/downsampling techniques)
## Code and Resources Used
- **Database:** AWS S3
- **Packages:** json, io, boto3, pandas, numpy, matplotlib, seaborn, datetime, sklearn, math, scipy, catboost, lightgbm, imblearn, hyperopt, xgboost, vecstack 
- Special thanks to **Guanlan** for Kalman Filter script
