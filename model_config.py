# model_config.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb

# ------------------- 模型配置 -------------------
ALL_MODELS_CONFIG = {
    'LR': {
        'model': LogisticRegression(max_iter=1000, random_state=42),
        'params': {
            'C': [0.001, 0.01, 0.1, 1],
            'penalty': ['l2'],
            'solver': ['liblinear'],
        },
        'needs_scaling': True
    },
    'Ridge': {
        'model': CalibratedClassifierCV(
            RidgeClassifier(random_state=42),
            method='sigmoid',  # 或 'isotonic'
            cv=3  # 用3折交叉验证校准
        ),
        'params': {
            # 注意：参数需要传递给内部 estimator，用 'estimator__alpha' 格式
            'estimator__alpha': [0.001, 0.01, 0.1, 1],
            # 也可以调 CalibratedClassifierCV 自身的参数，如 cv
        },
        # 'needs_scaling': True
        'needs_scaling': False
    },
    'RF': {  # 新增随机森林
        'model': RandomForestClassifier(random_state=42, oob_score=True),
        'params': {
            'n_estimators': [50, 100],
            'max_depth': [3, 5, 7],
            'min_samples_split': [10, 20, 30],
            'min_samples_leaf': [5, 10, 15],
            'max_features': ['sqrt', 'log2'],
            'bootstrap': [True]
        },
        'needs_scaling': False
    },
    'MLP': {
        'model': MLPClassifier(max_iter=500, random_state=42, early_stopping=True,
                               validation_fraction=0.2, n_iter_no_change=10),
        'params': {
            'hidden_layer_sizes': [(30,), (50,)],
            'activation': ['relu'],
            'alpha': [0.01, 0.1, 0.5],
            'solver': ['adam'],
            'learning_rate_init': [0.001, 0.005]
        },
        'needs_scaling': True
    },
    'DT': {
        'model': DecisionTreeClassifier(random_state=42, ccp_alpha=0.01),
        'params': {
            'max_depth': [2, 3, 5],
            'min_samples_split': [5, 10, 15],
            'min_samples_leaf': [2, 4, 6],
            'max_features': ['sqrt']
        },
        'needs_scaling': False
    },
    'SVM': {
        'model': SVC(probability=True, random_state=42, max_iter=1000),
        'params': {
            'C': [0.001, 0.01, 0.1],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale'],
        },
        'needs_scaling': True
    },
    'KNN': {
        'model': KNeighborsClassifier(n_jobs=-1),
        'params': {
            'n_neighbors': [10, 15, 20],
            'weights': ['uniform'],
            'metric': ['euclidean']
        },
        'needs_scaling': True
    },
    'XGBoost': {
        'model': xgb.XGBClassifier(random_state=42, eval_metric='auc'),
        'params': {
            'n_estimators': [5, 10],
            'max_depth': [2, 3],
            'learning_rate': [0.001, 0.005],
            'reg_alpha': [1, 2, 5],
            'reg_lambda': [5, 10, 20],
            'gamma': [0.5, 1, 2],
            'subsample': [0.6, 0.7, 0.8],
        },
        'needs_scaling': False
    }
}

# ------------------- 数据列定义 -------------------
# 临床特征（示例，请根据实际修改）
CLINICAL_FEATURES = ['Age', 'Gender', 'BMI', 'Smoking_Status']
PRS_FEATURES = ['PRS']
# SNP_FEATURES = ['rs16969968', 'rs1051730', 'rs8034191', 'rs1800925', 'rs1800629']  # 示例

# 特征组合（供 subgroup_analysis 使用）
FEATURE_COMBINATIONS = {
    # 'Age': ['Age'],
    # 'Gender': ['Gender'],
    # 'BMI': ['BMI'],
    # 'Smoking_Status': ['Smoking_Status'],
    # 'FVC_Pred': ['FVC_Pred'],
    # 'FEV1_Pred': ['FEV1_Pred'],
    # 'PRS': ['PRS'],
    # 'Age + Gender + BMI + Smoking_Status': CLINICAL_FEATURES,
    # 'Age + Gender + BMI + Smoking_Status + PRS': CLINICAL_FEATURES + PRS_FEATURES,

    'PRS Model': ['PRS'],
    'Clinical Feature Model': CLINICAL_FEATURES,
    'Integrated Risk Assessment Model': CLINICAL_FEATURES + PRS_FEATURES,

    # 'Clinical+SNP': CLINICAL_FEATURES + SNP_FEATURES,
    # 'All': CLINICAL_FEATURES + PRS_FEATURES + SNP_FEATURES,
}

# 分层条件（用于 subgroup_analysis）
SUBGROUP_CONDITIONS = {
    'Smoking_never': {'Smoking_Status': ['never']},
    # 'Age_ge65': {'Age_Cat4': ['>=65']},
    # 'PRS_high': {'PRS_cat': ['high']},
}

# 从分层条件中提取所有分层列名
STRAT_COLS = list(set(
    col for cond in SUBGROUP_CONDITIONS.values() for col in cond.keys()
))

# 目标列、样本ID列、验证集标记列
TARGET_COL = 'IsCase'
# 目标标签映射
POSITIVE_LABEL = 'Yes'   # 病例
NEGATIVE_LABEL = 'No'    # 对照
SAMPLE_ID_COL = 'ID'
VAL_FLAG_COL = 'IsInValSet'   # Yes 表示测试集，No 表示训练集
