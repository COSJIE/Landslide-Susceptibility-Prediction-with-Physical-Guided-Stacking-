# Landslide Susceptibility Prediction with TRIGRS & Stacking Deep Learning
﻿
## Overview
The code implements:
- Data preprocessing & feature engineering (including 5 groups of physically-guided interaction features)
- TRIGRS-derived FOS processing
- Feature selection (mutual information, top 15)
- Stacking model (GRU + Transformer + SVM meta-learner)
- Bayesian hyperparameter optimization
- 5-fold cross-validation
- Model evaluation (AUC-ROC, Spearman correlation, SHAP)
- Susceptibility zoning & Table 3 statistics
﻿
## Requirements
Python 3.8+  
Install dependencies:
```bash
conda install -r requirements.txt
After downloading the installation package. Enter the data with the name 'original data' into slide_prediction_2-dynamic_stockingGRU_Svm2.py to run the result.
