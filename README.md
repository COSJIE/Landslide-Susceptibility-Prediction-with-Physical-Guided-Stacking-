Integrating TRIGRS-Based Physical Labels with Stacking Deep Learning for Landslide Susceptibility Prediction
The study proposes a hybrid Stacking deep learning framework that integrates physically-informed labels derived from the TRIGRS model (factor of safety, FOS) with Gated Recurrent Unit (GRU), Transformer, and Support Vector Machine (SVM)
├── data/                      # Sample or preprocessed data (full raw data available upon request due to size/privacy)
│   ├── raw/                   # Original geospatial data (if included)
│   ├── processed/             # Preprocessed features, FOS labels, and susceptibility index
│   └── landslide_inventory.csv  # Historical landslide points (anonymized/sampled)
├── src/                       # Main source code
│   ├── preprocessing.py       # Data cleaning, feature engineering (including 5 groups of physically-guided interaction features)
│   ├── trigrs_processing.py   # Scripts for running TRIGRS and extracting quasi-steady FOS
│   ├── feature_selection.py   # Mutual information-based feature selection (top 15 features)
│   ├── model_training.py      # Stacking model implementation (GRU + Transformer + SVM meta-learner)
│   ├── bayesian_optimization.py # Hyperparameter tuning with Bayesian optimization
│   ├── cross_validation.py    # 5-fold cross-validation for meta-feature generation
│   ├── evaluation.py          # AUC-ROC, Spearman correlation, SHAP analysis, susceptibility zoning
│   └── utils.py               # Helper functions (e.g., weighted loss, pseudo-labeling)
├── notebooks/                 # Jupyter notebooks for exploratory analysis and result visualization
│   └── main_analysis.ipynb    # End-to-end reproduction notebook
├── results/                   # Output files (figures, tables, SHAP plots, zoning maps)
│   ├── figures/               # SHAP beeswarm/bar plots, susceptibility zoning map (Fig. 17)
│   └── tables/                # Table 3 (area ratio, disaster ratio, frequency ratio)
├── requirements.txt           # Python dependencies
├── environment.yml            # Optional Conda environment file
├── README.md                  # This file
└── LICENSE                    # MIT License (or specify your preferred license)
