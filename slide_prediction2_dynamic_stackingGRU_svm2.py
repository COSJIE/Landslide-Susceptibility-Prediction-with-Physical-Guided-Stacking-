import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_selection import mutual_info_classif
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore, spearmanr
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.svm import SVC
from bayes_opt import BayesianOptimization
import pickle
import os
import warnings

warnings.filterwarnings('ignore')

# Dependency check
required_modules = {
    'pandas': 'pandas',
    'numpy': 'numpy',
    'sklearn': 'scikit-learn',
    'seaborn': 'seaborn',
    'matplotlib': 'matplotlib',
    'scipy': 'scipy',
    'torch': 'torch (use: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu)',
    'bayes_opt': 'bayesian-optimization'
}
for module, install_name in required_modules.items():
    try:
        __import__(module)
    except ImportError:
        print(f"Module {module} not found. Install it using: pip install {install_name}")
        sys.exit(1)

print(f"PyTorch version: {torch.__version__}")
print(f"Python version: {sys.version}")

# Data file path
data_path = 'Data file path/original_data.xlsx'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Data file {data_path} not found")

# preprocessing
print("=== Start preprocessing ===")
df = pd.read_excel(data_path)

# Handling FOS Outliers
fos_outliers = df[df['fos'] >= 10]
print("FOS outliers:\n", fos_outliers[['FID', 'fos', 'CID']])
df.loc[df['fos'] >= 10, 'fos'] = df['fos'].median()

# Detect other outliers
numeric_cols = df.select_dtypes(include=np.number).columns.drop(['FID', 'CID'])
z_scores = df[numeric_cols].apply(zscore)
outliers = (z_scores.abs() > 3).any(axis=1)
print("Number of other outlier samples:", outliers.sum())
for col in numeric_cols:
    df.loc[z_scores[col].abs() > 3, col] = df[col].median()

# lithology geological weight
df['lithology'] = df['lithology'].map({1: 3, 2: 2, 3: 1})

# Construct interaction terms
df['Slope_rainfall'] = df['Slope'] * df['rainfall']
df['DoR_TWI'] = df['DoR'] * df['TWI']
df['Slope_SPI'] = df['Slope'] * df['SPI']
df['Elevation_rainfall'] = df['Elevation'] * df['rainfall']
df['fos_lithology'] = df['fos'] * df['lithology']
df['is_fos_low'] = (df['fos'] < 1.1).astype(int)

X = df.drop(columns=['FID', 'CID'])
y = df['CID']

# Data splitting
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=2 / 9, stratify=y_temp, random_state=42)
print("Number of training samples:", len(X_train))
print("Number of validation samples:", len(X_val))
print("Number of test samples:", len(X_test))

# Standardization
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

fos_train = X_train['fos'].values
fos_val = X_val['fos'].values
fos_test = X_test['fos'].values

# Feature selection: top 15 features by mutual information
mi_scores = mutual_info_classif(X_train_scaled, y_train, random_state=42)
feature_names = X.columns
mi_scores_df = pd.DataFrame({'Feature': feature_names, 'MI_Score': mi_scores})
mi_scores_df = mi_scores_df.sort_values(by='MI_Score', ascending=False)
print("Feature mutual information scores:")
for _, row in mi_scores_df.iterrows():
    print(f"{row['Feature']}: {row['MI_Score']:.4f}")
mi_scores_df.to_csv('mi_scores.csv', index=False, encoding='utf-8')

# Keep top 15 features, force keep key features
top_k = 15
selected_features = mi_scores_df['Feature'].head(top_k).values
key_features = ['is_fos_low', 'fos']

for i, key_feature in enumerate(key_features):
    if key_feature not in selected_features:
        lowest_non_key = mi_scores_df[~mi_scores_df['Feature'].isin(key_features)]['Feature'].iloc[top_k - len(key_features) + i]
        selected_features = np.where(selected_features == lowest_non_key, key_feature, selected_features)

assert len(selected_features) == top_k, f"Feature count mismatch: expected {top_k}, got {len(selected_features)}"

selected_features_mask = np.isin(feature_names, selected_features)
X_train_scaled = X_train_scaled[:, selected_features_mask]
X_val_scaled = X_val_scaled[:, selected_features_mask]
X_test_scaled = X_test_scaled[:, selected_features_mask]
print("Selected features after filtering:", selected_features)

# Save processed data
np.savez('processed_data.npz', X_train_scaled=X_train_scaled, X_val_scaled=X_val_scaled, X_test_scaled=X_test_scaled,
         y_train=y_train.values, y_val=y_val.values, y_test=y_test.values, fos_train=fos_train, fos_val=fos_val,
         fos_test=fos_test, selected_features=selected_features)

# Save selected features
with open('selected_features.txt', 'w', encoding='utf-8') as f:
    f.write("Selected features after filtering:\n")
    for feature in selected_features:
        f.write(f"{feature}\n")

# ------------------ 2. Visualization ------------------
print("\n=== Start visualization ===")
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['fos'], y=df['CID'], hue=df['CID'])
plt.axvline(x=1.1, color='red', linestyle='--')
plt.savefig('fos_cid_scatter.png')
plt.close()

# ------------------ 3. Model definition ------------------
weights = np.where(fos_train < 1.1, 6.0 / (fos_train + 0.1), 1.0)

class GRUModel(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out.squeeze(-1))

class GRUWrapper:
    def __init__(self, model, epochs=50, batch_size=32, device='cpu'):
        self.model = model.to(device)
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y, validation_data=None, sample_weight=None):
        X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)
        else:
            sample_weight = torch.ones_like(y)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss(reduction='none')

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                batch_weight = sample_weight[i:i + self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss = (loss * batch_weight).mean()
                loss.backward()
                optimizer.step()

            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    X_val, y_val = validation_data
                    X_val = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1).to(self.device)
                    y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    outputs = self.model(X_val)
                    val_loss = criterion(outputs, y_val).mean().item()
                print(f"Epoch {epoch + 1}/{self.epochs}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'gru_best.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping")
                        break

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            X = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            proba = self.model(X).cpu().numpy()
        return np.hstack((1 - proba.reshape(-1, 1), proba.reshape(-1, 1)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

class TransformerModel(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.5):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, d_model)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True),
            num_layers
        )
        self.fc = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.input_layer(x).unsqueeze(1)
        x = self.transformer_encoder(x)
        x = self.fc(x[:, -1, :]).squeeze(-1)
        return self.sigmoid(x)

class PyTorchModelWrapper:
    def __init__(self, model, pseudo_labels, alpha=0.3, epochs=50, batch_size=32, device='cpu'):
        self.model = model.to(device)
        self.pseudo_labels = torch.tensor(pseudo_labels, dtype=torch.float32).to(device)
        self.alpha = alpha
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

    def fit(self, X, y, validation_data=None, sample_weight=None, pseudo_labels_val=None):
        X = torch.tensor(X, dtype=torch.float32).to(self.device)
        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        if sample_weight is not None:
            sample_weight = torch.tensor(sample_weight, dtype=torch.float32).to(self.device)
        else:
            sample_weight = torch.ones_like(y)

        optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        criterion = nn.BCELoss(reduction='none')

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model.train()
            for i in range(0, len(X), self.batch_size):
                batch_X = X[i:i + self.batch_size]
                batch_y = y[i:i + self.batch_size]
                batch_pseudo = self.pseudo_labels[i:i + self.batch_size]
                batch_weight = sample_weight[i:i + self.batch_size]

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                ce_loss = criterion(outputs, batch_y)
                pseudo_loss = criterion(outputs, batch_pseudo)
                loss = (1 - self.alpha) * ce_loss + self.alpha * pseudo_loss
                loss = (loss * batch_weight).mean()
                loss.backward()
                optimizer.step()

            if validation_data:
                self.model.eval()
                with torch.no_grad():
                    X_val, y_val = validation_data
                    X_val = torch.tensor(X_val, dtype=torch.float32).to(self.device)
                    y_val = torch.tensor(y_val, dtype=torch.float32).to(self.device)
                    if pseudo_labels_val is not None:
                        pseudo_val = torch.tensor(pseudo_labels_val, dtype=torch.float32).to(self.device)
                    else:
                        pseudo_val = torch.zeros_like(y_val)
                    outputs = self.model(X_val)
                    ce_loss = criterion(outputs, y_val)
                    pseudo_loss = criterion(outputs, pseudo_val)
                    val_loss = ((1 - self.alpha) * ce_loss + self.alpha * pseudo_loss).mean().item()
                print(f"Epoch {epoch + 1}/{self.epochs}, Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), 'transformer_best.pth')
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= 10:
                        print("Early stopping")
                        break

    def predict_proba(self, X):
        self.model.eval()
        with torch.no_grad():
            proba = self.model(torch.tensor(X, dtype=torch.float32).to(self.device)).cpu().numpy()
        return np.hstack((1 - proba.reshape(-1, 1), proba.reshape(-1, 1)))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

# ------------------ 4. Bayesian Optimization ------------------
print("\n=== Bayesian Optimization ===")

def compute_pseudo_labels(fos_values):
    fos_values = np.array(fos_values)
    pseudo = 1 / (1 + np.exp(4 * (fos_values - 1.1)))
    return pseudo

pseudo_labels_train = compute_pseudo_labels(fos_train)
pseudo_labels_val = compute_pseudo_labels(fos_val)

def generate_stacking_features(X, y, fos, pseudo_labels, gru_params, transformer_params, n_folds=5):
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    gru_probas = np.zeros(len(y))
    transformer_probas = np.zeros(len(y))

    weights = np.where(fos < 1.1, 6.0 / (fos + 0.1), 1.0)

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"Processing fold {fold + 1}...")
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        weights_train = weights[train_idx]
        pseudo_train = pseudo_labels[train_idx]
        pseudo_val = pseudo_labels[val_idx]

        # Train GRU
        gru_model = GRUModel(**gru_params)
        gru_wrapper = GRUWrapper(gru_model, device='cuda' if torch.cuda.is_available() else 'cpu')
        gru_wrapper.fit(X_train, y_train, (X_val, y_val), sample_weight=weights_train)
        gru_probas[val_idx] = gru_wrapper.predict_proba(X_val)[:, 1]

        # Train Transformer
        transformer_model = TransformerModel(**transformer_params)
        transformer_wrapper = PyTorchModelWrapper(
            transformer_model, pseudo_train, alpha=0.3, device='cuda' if torch.cuda.is_available() else 'cpu')
        transformer_wrapper.fit(X_train, y_train, (X_val, y_val), sample_weight=weights_train,
                                pseudo_labels_val=pseudo_val)
        transformer_probas[val_idx] = transformer_wrapper.predict_proba(X_val)[:, 1]

    stacking_inputs = np.column_stack((gru_probas, transformer_probas))
    return stacking_inputs, y

def objective(hidden_dim, num_layers, learning_rate, d_model, nhead, num_layers_transformer, dim_feedforward,
              svm_log_C, svm_log_gamma):
    gru_params = {
        'input_dim': X_train_scaled.shape[1],
        'hidden_dim': int(hidden_dim),
        'num_layers': int(num_layers),
        'dropout': 0.5
    }

    d_model = int(d_model)
    nhead = int(nhead)
    possible_nheads = [i for i in range(2, d_model + 1) if d_model % i == 0]
    if not possible_nheads:
        nhead = 2
    else:
        nhead = min(possible_nheads, key=lambda x: abs(x - nhead))

    transformer_params = {
        'input_dim': X_train_scaled.shape[1],
        'd_model': d_model,
        'nhead': nhead,
        'num_layers': int(num_layers_transformer),
        'dim_feedforward': int(dim_feedforward),
        'dropout': 0.5
    }

    stacking_inputs, stacking_targets = generate_stacking_features(
        X_val_scaled, y_val.values, fos_val, pseudo_labels_val, gru_params, transformer_params)

    meta_model = SVC(
        C=10 ** svm_log_C,
        gamma=10 ** svm_log_gamma,
        kernel='rbf',
        probability=True,
        random_state=42
    )
    meta_model.fit(stacking_inputs, stacking_targets)

    stacking_proba = meta_model.predict_proba(stacking_inputs)[:, 1]
    auc = roc_auc_score(stacking_targets, stacking_proba)
    return auc

pbounds = {
    'hidden_dim': (32, 128),
    'num_layers': (1, 3),
    'learning_rate': (0.001, 0.01),
    'd_model': (16, 64),
    'nhead': (2, 8),
    'num_layers_transformer': (1, 3),
    'dim_feedforward': (32, 128),
    'svm_log_C': (-1, 1),
    'svm_log_gamma': (-2, 0)
}

optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=42)
optimizer.maximize(init_points=5, n_iter=15)

best_params = optimizer.max['params']
print(f"Best parameters: {best_params}, AUC: {optimizer.max['target']:.4f}")

# Train final models
gru_best_params = {
    'input_dim': X_train_scaled.shape[1],
    'hidden_dim': int(best_params['hidden_dim']),
    'num_layers': int(best_params['num_layers']),
    'dropout': 0.5
}
gru_best_model = GRUModel(**gru_best_params)
gru_best_wrapper = GRUWrapper(gru_best_model, device='cuda' if torch.cuda.is_available() else 'cpu')
gru_best_wrapper.fit(X_train_scaled, y_train.values, (X_val_scaled, y_val.values), sample_weight=weights)

d_model = int(best_params['d_model'])
possible_nheads = [i for i in range(2, d_model + 1) if d_model % i == 0]
nhead = min(possible_nheads, key=lambda x: abs(x - int(best_params['nhead'])))

transformer_best_params = {
    'input_dim': X_train_scaled.shape[1],
    'd_model': d_model,
    'nhead': nhead,
    'num_layers': int(best_params['num_layers_transformer']),
    'dim_feedforward': int(best_params['dim_feedforward']),
    'dropout': 0.5
}
transformer_best_model = TransformerModel(**transformer_best_params)
pytorch_best_model = PyTorchModelWrapper(
    transformer_best_model, pseudo_labels_train, alpha=0.3, device='cuda' if torch.cuda.is_available() else 'cpu')
pytorch_best_model.fit(X_train_scaled, y_train.values, (X_val_scaled, y_val.values), sample_weight=weights,
                       pseudo_labels_val=pseudo_labels_val)

# Generate Stacking features and train meta model
print("\n=== Generating Stacking features ===")
stacking_inputs, stacking_targets = generate_stacking_features(
    X_val_scaled, y_val.values, fos_val, pseudo_labels_val, gru_best_params, transformer_best_params)

meta_model = SVC(
    C=10 ** best_params['svm_log_C'],
    gamma=10 ** best_params['svm_log_gamma'],
    kernel='rbf',
    probability=True,
    random_state=42
)
meta_model.fit(stacking_inputs, stacking_targets)
print("Stacking meta model (SVM) training completed")

# Test set prediction using Stacking
gru_proba = gru_best_wrapper.predict_proba(X_test_scaled)[:, 1]
transformer_proba = pytorch_best_model.predict_proba(X_test_scaled)[:, 1]
stacking_test_inputs = np.column_stack((gru_proba, transformer_proba))
stacking_proba = meta_model.predict_proba(stacking_test_inputs)[:, 1]

# Post-processing adjustment
stacking_pred = np.zeros_like(stacking_proba, dtype=int)
for i, fos in enumerate(fos_test):
    if fos < 1.1:
        stacking_pred[i] = 1 if stacking_proba[i] >= 0.4 else 0
    else:
        stacking_pred[i] = 1 if stacking_proba[i] >= 0.5 else 0

# ------------------ 5. Evaluation ------------------
print("\n=== Model Evaluation ===")

class StackingModel:
    def __init__(self, proba, pred, gru_wrapper, transformer_wrapper, meta_model):
        self.proba = proba.reshape(-1, 1)
        self.pred = pred
        self.gru_wrapper = gru_wrapper
        self.transformer_wrapper = transformer_wrapper
        self.meta_model = meta_model

    def predict_proba(self, X):
        return np.hstack((1 - self.proba, self.proba))

    def predict(self, X):
        return self.pred

from sklearn.metrics import roc_curve, auc

def evaluate_model(model, X, y, fos, name):
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]
    print(f"\n=== {name} ===")
    print("Classification report:\n", classification_report(y, y_pred))
    print("AUC:", roc_auc_score(y, y_pred_proba))
    print("Confusion matrix:\n", confusion_matrix(y, y_pred))

    fos_unstable = fos < 1.1
    if np.any(fos_unstable):
        print("Proportion predicted as CID=1 when FOS < 1.1:", np.mean(y_pred[fos_unstable]))
    fos_stable = fos >= 1.1
    if np.any(fos_stable):
        print("Proportion predicted as CID=0 when FOS >= 1.1:", np.mean(1 - y_pred[fos_stable]))

    corr, p_value = spearmanr(fos, y_pred_proba)
    print(f"Spearman correlation between FOS and predicted probability: {corr:.4f} (p-value: {p_value:.4f})")

    # Save FOS and predicted probability to CSV
    fos_proba_df = pd.DataFrame({
        'FOS': fos,
        'Probability': y_pred_proba,
        'True_Label': y
    })
    fos_proba_path = os.path.join(os.getcwd(), f'fos_pred_proba_{name.lower()}.csv')
    fos_proba_df.to_csv(fos_proba_path, index=False, encoding='utf-8')
    print(f"FOS, predicted probability and true labels saved to {fos_proba_path}")

    # Compute and save ROC curve data
    fpr, tpr, thresholds = roc_curve(y, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    roc_df = pd.DataFrame({
        'FPR': fpr,
        'TPR': tpr,
        'Threshold': thresholds
    })
    roc_df.to_csv(f'roc_curve_{name.lower()}.csv', index=False, encoding='utf-8')
    print(f"ROC curve data saved to roc_curve_{name.lower()}.csv (AUC: {roc_auc:.4f})")

    # Plot and save ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {name}')
    plt.legend(loc='lower right')
    plt.savefig(f'roc_curve_{name.lower()}.png')
    plt.close()
    print(f"ROC curve plot saved to roc_curve_{name.lower()}.png")

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=fos, y=y_pred_proba, hue=y, style=y)
    plt.axvline(x=1.1, color='red', linestyle='--')
    plt.xlabel('FOS')
    plt.ylabel('Predicted Probability (CID=1)')
    plt.title(f'{name}: FOS vs Predicted Probability')
    plt.savefig(f'fos_pred_proba_{name.lower()}.png')
    plt.close()

stacking_model = StackingModel(stacking_proba, stacking_pred, gru_best_wrapper, pytorch_best_model, meta_model)

evaluate_model(gru_best_wrapper, X_test_scaled, y_test, fos_test, "GRU")
evaluate_model(pytorch_best_model, X_test_scaled, y_test, fos_test, "Transformer")
evaluate_model(stacking_model, X_test_scaled, y_test, fos_test, "Stacking")

# SHAP analysis
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# Set global font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.unicode_minus'] = False


def compute_shap_values(model, X, feature_names, output_path='shap_values.csv', plot_path='shap_summary.png'):
    def stacking_predict(X):
        X_tensor = torch.tensor(X, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
        gru_proba = model.gru_wrapper.predict_proba(X)[:, 1]
        transformer_proba = model.transformer_wrapper.predict_proba(X)[:, 1]
        stacking_inputs = np.column_stack((gru_proba, transformer_proba))
        return model.meta_model.predict_proba(stacking_inputs)[:, 1]

    explainer = shap.KernelExplainer(stacking_predict, X)
    shap_values = explainer.shap_values(X)

    shap_df = pd.DataFrame(shap_values, columns=feature_names)
    shap_df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"SHAP values saved to {output_path}")

    # Compute and save feature importance (mean absolute SHAP)
    feature_importance = np.abs(shap_df).mean().sort_values(ascending=False)
    importance_df = pd.DataFrame({
        'Feature': feature_importance.index,
        'Mean_Abs_SHAP': feature_importance.values
    })
    importance_df.to_csv('shap_feature_importance.csv', index=False, encoding='utf-8')
    print("SHAP feature importance saved to shap_feature_importance.csv")

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title('SHAP Feature Importance Analysis', fontfamily='Times New Roman')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"SHAP summary plot saved to {plot_path}")


# Add 5-fold cross-validation after model evaluation section
print("\n=== 5-fold Cross-Validation Results ===")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_auc_scores = []
cv_spearman_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
    print(f"\nFold {fold + 1}/5:")

    # Split data
    X_cv_train = X_train_scaled[train_idx]
    y_cv_train = y_train.values[train_idx]
    X_cv_val = X_train_scaled[val_idx]
    y_cv_val = y_train.values[val_idx]
    fos_cv_val = fos_train[val_idx]

    # Train GRU model
    gru_cv = GRUModel(**gru_best_params)
    gru_wrapper_cv = GRUWrapper(gru_cv, device='cuda' if torch.cuda.is_available() else 'cpu')

    # Compute pseudo labels
    pseudo_cv = compute_pseudo_labels(fos_train[train_idx])

    # Train Transformer model
    transformer_cv = TransformerModel(**transformer_best_params)
    transformer_wrapper_cv = PyTorchModelWrapper(
        transformer_cv, pseudo_cv, alpha=0.3,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Train base models
    gru_wrapper_cv.fit(X_cv_train, y_cv_train)
    transformer_wrapper_cv.fit(X_cv_train, y_cv_train)

    # Predict
    gru_proba_cv = gru_wrapper_cv.predict_proba(X_cv_val)[:, 1]
    transformer_proba_cv = transformer_wrapper_cv.predict_proba(X_cv_val)[:, 1]

    # Stacking features
    stacking_inputs_cv = np.column_stack((gru_proba_cv, transformer_proba_cv))

    # Train SVM meta model
    svm_cv = SVC(
        C=10 ** best_params['svm_log_C'],
        gamma=10 ** best_params['svm_log_gamma'],
        kernel='rbf',
        probability=True,
        random_state=42
    )
    svm_cv.fit(stacking_inputs_cv, y_cv_val)

    # Final prediction
    stacking_proba_cv = svm_cv.predict_proba(stacking_inputs_cv)[:, 1]

    # Compute metrics
    auc_cv = roc_auc_score(y_cv_val, stacking_proba_cv)
    spearman_cv, _ = spearmanr(fos_cv_val, stacking_proba_cv)

    cv_auc_scores.append(auc_cv)
    cv_spearman_scores.append(spearman_cv)

    print(f"  AUC: {auc_cv:.4f}, Spearman correlation: {spearman_cv:.4f}")

# Summary
print("\n=== Cross-Validation Summary ===")
print(f"Mean AUC: {np.mean(cv_auc_scores):.4f} (±{np.std(cv_auc_scores):.4f})")
print(f"Mean Spearman correlation: {np.mean(cv_spearman_scores):.4f} (±{np.std(cv_spearman_scores):.4f})")
print(f"Range of AUC: {np.min(cv_auc_scores):.4f} - {np.max(cv_auc_scores):.4f}")

# Save results
cv_results = pd.DataFrame({
    'Fold': range(1, 6),
    'AUC': cv_auc_scores,
    'Spearman': cv_spearman_scores
})
cv_results.to_csv('cross_validation_results.csv', index=False, encoding='utf-8')
print("Cross-validation results saved to cross_validation_results.csv")

# Plot CV results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.boxplot(cv_auc_scores)
plt.title('AUC across 5 Folds')
plt.ylabel('AUC')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.boxplot(cv_spearman_scores)
plt.title('Spearman Correlation across 5 Folds')
plt.ylabel('Spearman ρ')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('cross_validation_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()
print("Cross-validation boxplot saved to cross_validation_boxplot.png")

print("\n=== Computing SHAP values ===")
compute_shap_values(stacking_model, X_test_scaled, selected_features)