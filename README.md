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

Put the data file into the corresponding execution path and start running it, with annotations for each step

Data contains 15 initial conditioning factors—elevation, slope, aspect, plan curvature, profile curvature, rainfall, surface roughness (SR), lithology, normalized difference vegetation index (NDVI), river core density (RICD), fault core density (FCD), road core density (ROCD), stream power index (SPI), topographic wetness index (TWI), and degree of relief (DoR)—along with a binary landslide label (1 = landslide, 0 = non-landslide) and the corresponding factor of safety (FOS) computed by TRIGRS.

 
## Requirements
Python 3.8+  
Install dependencies:
```bash
conda install -r requirements.txt
After downloading the installation package. Enter the data with the name 'original data' into slide_prediction_2-dynamic_stockingGRU_Svm2.py to run the result.

