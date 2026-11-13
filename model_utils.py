# utils.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.linewidth'] = 4
plt.rcParams['xtick.labelsize'] = 18
plt.rcParams['ytick.labelsize'] = 18
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, roc_curve, confusion_matrix,
                             accuracy_score, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
import shap
import warnings
from model_config import POSITIVE_LABEL, NEGATIVE_LABEL
warnings.filterwarnings('ignore')

# ------------------- 数据加载与预处理 -------------------
def load_data(file_path, sheet_name=None):
    """加载Excel或CSV数据"""
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path, comment='#')
    elif file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path, sheet_name=sheet_name or 0, comment='#')
    else:
        df = pd.read_csv(file_path, sep='\t', comment='#')
    return df


def preprocess_data(df, target_col, feature_cols, strat_cols, sample_id_col='ID'):
    """
    预处理数据：保留指定特征、编码分类变量、处理缺失值、返回特征矩阵和标签
    同时为分层列保存原始值（以 _orig 后缀）
    """
    # 复制数据
    df_proc = df.copy()
    
    # 为分层列创建原始副本
    for col in strat_cols:
        if col in df_proc.columns:
            df_proc[f'{col}_orig'] = df_proc[col].copy()
    
    # 确保目标列存在
    if target_col not in df_proc.columns:
        raise ValueError(f"目标列 {target_col} 不存在")
    
    # 删除目标列缺失的行
    df_proc = df_proc.dropna(subset=[target_col])
    
    # 检查特征是否存在
    missing_feats = [f for f in feature_cols if f not in df_proc.columns]
    if missing_feats:
        print(f"警告：以下特征缺失，将被忽略：{missing_feats}")
        feature_cols = [f for f in feature_cols if f in df_proc.columns]
    
    if not feature_cols:
        raise ValueError("无有效特征")
    
    # 删除特征缺失的行
    df_proc = df_proc.dropna(subset=feature_cols)
    
    # 编码分类特征
    X = df_proc[feature_cols].copy()
    cat_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        X[col] = le.fit_transform(X[col].astype(str))

    # 目标标签映射
    y = df_proc[target_col].map({POSITIVE_LABEL: 1, NEGATIVE_LABEL: 0}).values
    if pd.isnull(y).any():
        raise ValueError(f"目标列 {target_col} 包含非 '{POSITIVE_LABEL}'/'{NEGATIVE_LABEL}' 的值")
    y = y.astype(int)
    
    # 返回特征、标签、原始分层数据（仅用于子组筛选）
    strat_df = df_proc[[f'{col}_orig' for col in strat_cols if f'{col}_orig' in df_proc.columns]]
    
    return X, y, strat_df, feature_cols


# ------------------- 评估指标（含Bootstrap置信区间） -------------------
def bootstrap_auc_ci(y_true, y_proba, n_bootstraps=1000, confidence_level=0.95):
    """计算AUC的Bootstrap置信区间"""
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    boot_auc = []
    rng = np.random.RandomState(42)
    for _ in range(n_bootstraps):
        idx = rng.choice(len(y_true), len(y_true), replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
        try:
            auc = roc_auc_score(y_true[idx], y_proba[idx])
            boot_auc.append(auc)
        except:
            continue
    if len(boot_auc) == 0:
        return np.nan, np.nan
    alpha = (1 - confidence_level) / 2
    lower = np.percentile(boot_auc, alpha*100)
    upper = np.percentile(boot_auc, (1-alpha)*100)
    return lower, upper


def calculate_metrics(y_true, y_proba, threshold=None):
    """
    计算分类指标（AUC、灵敏度、特异度等），若threshold=None则使用约登指数确定最佳阈值
    返回字典
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    if threshold is None:
        youden = tpr - fpr
        best_idx = np.argmax(youden)
        threshold = thresholds[best_idx]
    
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp+fn) else np.nan
    specificity = tn / (tn + fp) if (tn+fp) else np.nan
    accuracy = accuracy_score(y_true, y_pred)
    ppv = tp / (tp + fp) if (tp+fp) else np.nan
    npv = tn / (tn + fn) if (tn+fn) else np.nan
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    auc_ci = bootstrap_auc_ci(y_true, y_proba)
    brier = brier_score_loss(y_true, y_proba)
    
    return {
        'threshold': threshold,
        'auc': auc,
        'auc_ci_lower': auc_ci[0],
        'auc_ci_upper': auc_ci[1],
        'sensitivity': sensitivity,
        'specificity': specificity,
        'accuracy': accuracy,
        'ppv': ppv,
        'npv': npv,
        'f1_score': f1,
        'brier': brier,
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'fpr': fpr, 'tpr': tpr,
    }


# ------------------- 决策曲线 -------------------
def decision_curve(y_true, y_proba, thresholds=None):
    """
    计算决策曲线的净收益
    thresholds: 概率阈值列表，默认为0.01到0.99
    返回: thresholds, net_benefit_model, net_benefit_all, net_benefit_none
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 50)
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    n = len(y_true)
    # 患病率
    prevalence = y_true.mean()
    nb_model = []
    nb_all = []
    nb_none = []
    
    for p in thresholds:
        y_pred = (y_proba >= p).astype(int)
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        # 净收益 = (TP/n) - (FP/n) * (p/(1-p))
        benefit = (tp / n) - (fp / n) * (p / (1 - p))
        nb_model.append(benefit)
        # 全部治疗策略
        nb_all.append(prevalence - (1-prevalence)*(p/(1-p)))
        # 全部不治疗策略
        nb_none.append(0)
    return thresholds, np.array(nb_model), np.array(nb_all), np.array(nb_none)


# ------------------- 绘图函数 -------------------
def plot_roc_curves(results_dict, title, save_path=None, width=9, height=8):
    """绘制多条ROC曲线，results_dict: {model_name: metrics_dict}"""
    plt.figure(figsize=(width, height))
    for name, metrics in results_dict.items():
        # plt.plot(metrics['fpr'], metrics['tpr'], lw=2,
        #          label=f"{name} (AUC={metrics['auc']:.3f}, 95% CI {metrics['auc_ci_lower']:.3f}-{metrics['auc_ci_upper']:.3f})")
        # plt.plot(metrics['fpr'], metrics['tpr'], lw=2,
        #          label=f"{name}: {metrics['auc']:.2f} ({metrics['auc_ci_lower']:.2f}-{metrics['auc_ci_upper']:.2f})")
        plt.plot(metrics['fpr'], metrics['tpr'], lw=5,
                         label=f"{metrics['auc']:.2f} ({metrics['auc_ci_lower']:.2f}-{metrics['auc_ci_upper']:.2f})")
    # plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess (AUC=0.5)')
    plt.plot([0, 1], [0, 1], 'k--', lw=5, label='0.5')
    plt.xlabel('1 - Specificity', fontsize=22)
    plt.ylabel('Sensitivity', fontsize=22)
    plt.title(title, fontsize=22)
    # plt.legend(loc='lower right', fontsize=12)
    plt.legend(loc='lower right', fontsize=20, title="AUC (95% CI)", alignment='left', title_fontsize=20)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_calibration_curves(y_true, proba_dict, save_path=None):
    """绘制校准曲线，proba_dict: {model_name: predicted_proba}"""
    plt.figure(figsize=(10,8))
    for name, proba in proba_dict.items():
        fraction_pos, mean_pred = calibration_curve(y_true, proba, n_bins=10)
        brier = brier_score_loss(y_true, proba)
        plt.plot(mean_pred, fraction_pos, 'o-', lw=5, label=f"{name} (Brier={brier:.3f})")
    plt.plot([0,1],[0,1], 'k--', lw=5, label='Perfectly Calibrated')
    plt.xlabel('Mean Predicted Probability', fontsize=22)
    plt.ylabel('Fraction of Positives', fontsize=22)
    plt.title('Calibration Curves', fontsize=22)
    plt.legend(fontsize=18)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


def plot_decision_curves(y_true, proba_dict, save_path=None):
    """绘制决策曲线，proba_dict: {model_name: predicted_proba}"""
    plt.figure(figsize=(10,8))
    thresholds = np.linspace(0.01, 0.99, 50)
    for name, proba in proba_dict.items():
        th, nb_model, nb_all, nb_none = decision_curve(y_true, proba, thresholds)
        # print("th:", th)
        # print("nb_all:", nb_all)
        # print("nb_none:", nb_none)
        plt.plot(th, nb_model, lw=5, label=name)
    # 添加参考线
    plt.plot(th, nb_all, 'k--', lw=5, label='Treat All')
    plt.plot(th, nb_none, 'k-', lw=5, label='Treat None')
    plt.xlabel('Threshold Probability', fontsize=22)
    plt.ylabel('Net Benefit', fontsize=22)
    plt.title('Decision Curves', fontsize=22)
    plt.legend(fontsize=20)
    plt.grid(alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


# ------------------- SHAP分析 -------------------
def compute_shap(model, X_train, model_name):
    """计算SHAP值（简化版）"""
    try:
        if model_name in ['LR', 'Ridge']:
            explainer = shap.LinearExplainer(model, X_train, feature_perturbation="interventional")
        elif model_name in ['RF', 'DT', 'XGBoost']:
            explainer = shap.TreeExplainer(model)
        else:
            # 使用 KernelExplainer 采样
            background = shap.sample(X_train, min(100, len(X_train)))
            explainer = shap.KernelExplainer(lambda x: model.predict_proba(x)[:, 1], background)
        shap_values = explainer.shap_values(X_train)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 二分类取正类
        return shap_values, explainer.expected_value
    except Exception as e:
        print(f"SHAP计算失败：{e}")
        return None, None


# ------------------- NRI 计算 -------------------
# def calculate_nri(y_true, proba_old, proba_new, thresholds):
#     """
#     计算 categorical NRI (净重分类指数) 在给定阈值列表下。
#     返回字典，包含每个阈值的 NRI 和各个分类统计。
#     """
#     y_true = np.array(y_true)
#     proba_old = np.array(proba_old)
#     proba_new = np.array(proba_new)
#     results = {}
#     for th in thresholds:
#         pred_old = (proba_old >= th).astype(int)
#         pred_new = (proba_new >= th).astype(int)
#         # 病例 (y_true == 1)
#         case_idx = (y_true == 1)
#         n_cases = np.sum(case_idx)
#         if n_cases > 0:
#             up_cases = np.sum((pred_old[case_idx] == 0) & (pred_new[case_idx] == 1))
#             down_cases = np.sum((pred_old[case_idx] == 1) & (pred_new[case_idx] == 0))
#             nri_events = (up_cases / n_cases) - (down_cases / n_cases)
#         else:
#             up_cases = down_cases = 0
#             nri_events = np.nan
#
#         # 对照 (y_true == 0)
#         control_idx = (y_true == 0)
#         n_controls = np.sum(control_idx)
#         if n_controls > 0:
#             down_controls = np.sum((pred_old[control_idx] == 1) & (pred_new[control_idx] == 0))
#             up_controls = np.sum((pred_old[control_idx] == 0) & (pred_new[control_idx] == 1))
#             nri_nonevents = (down_controls / n_controls) - (up_controls / n_controls)
#         else:
#             down_controls = up_controls = 0
#             nri_nonevents = np.nan
#
#         nri = nri_events + nri_nonevents if not np.isnan(nri_events) and not np.isnan(nri_nonevents) else np.nan
#
#         results[th] = {
#             'threshold': th,
#             'nri': nri,
#             'nri_events': nri_events,
#             'nri_nonevents': nri_nonevents,
#             'up_cases': up_cases,
#             'down_cases': down_cases,
#             'up_controls': up_controls,
#             'down_controls': down_controls,
#             'n_cases': n_cases,
#             'n_controls': n_controls
#         }
#     return results


# ------------------- NRI 计算 -------------------
def calculate_nri(y_true, proba_old, proba_new, thresholds, n_bootstraps=1000, alpha=0.05):
    """
    计算 categorical NRI (净重分类指数) 在给定阈值列表下，
    并同时计算 NRI 的 Bootstrap 置信区间以及完整的重新分类表格。

    参数:
        y_true : array-like, 真实标签 (0/1)
        proba_old : array-like, 旧模型的预测概率
        proba_new : array-like, 新模型的预测概率
        thresholds : list of float, 风险阈值列表
        n_bootstraps : int, Bootstrap 重采样次数
        alpha : float, 显著性水平，默认 0.05 对应 95% CI

    返回:
        results : dict, 每个阈值对应一个子字典，包含:
            - nri, nri_events, nri_nonevents (点估计)
            - up_cases, down_cases, up_controls, down_controls
            - stay_low_cases, stay_high_cases, stay_low_controls, stay_high_controls
            - n_cases, n_controls
            - nri_ci_lower, nri_ci_upper (总NRI的置信区间)
            - nri_events_ci_lower, nri_events_ci_upper
            - nri_nonevents_ci_lower, nri_nonevents_ci_upper
    """
    y_true = np.array(y_true)
    proba_old = np.array(proba_old)
    proba_new = np.array(proba_new)
    results = {}

    for th in thresholds:
        pred_old = (proba_old >= th).astype(int)
        pred_new = (proba_new >= th).astype(int)

        # ---- 1. 计算点估计和原始分类计数 ----
        # 病例 (y_true == 1)
        case_idx = (y_true == 1)
        n_cases = np.sum(case_idx)
        if n_cases > 0:
            up_cases = np.sum((pred_old[case_idx] == 0) & (pred_new[case_idx] == 1))
            down_cases = np.sum((pred_old[case_idx] == 1) & (pred_new[case_idx] == 0))
            stay_low_cases = np.sum((pred_old[case_idx] == 0) & (pred_new[case_idx] == 0))
            stay_high_cases = np.sum((pred_old[case_idx] == 1) & (pred_new[case_idx] == 1))
            nri_events = (up_cases - down_cases) / n_cases
        else:
            up_cases = down_cases = stay_low_cases = stay_high_cases = 0
            nri_events = np.nan

        # 对照 (y_true == 0)
        control_idx = (y_true == 0)
        n_controls = np.sum(control_idx)
        if n_controls > 0:
            up_controls = np.sum((pred_old[control_idx] == 0) & (pred_new[control_idx] == 1))
            down_controls = np.sum((pred_old[control_idx] == 1) & (pred_new[control_idx] == 0))
            stay_low_controls = np.sum((pred_old[control_idx] == 0) & (pred_new[control_idx] == 0))
            stay_high_controls = np.sum((pred_old[control_idx] == 1) & (pred_new[control_idx] == 1))
            nri_nonevents = (down_controls - up_controls) / n_controls
        else:
            up_controls = down_controls = stay_low_controls = stay_high_controls = 0
            nri_nonevents = np.nan

        nri = nri_events + nri_nonevents if not np.isnan(nri_events) and not np.isnan(nri_nonevents) else np.nan

        # ---- 2. Bootstrap 计算置信区间 ----
        n = len(y_true)
        boot_nri = []
        boot_nri_events = []
        boot_nri_nonevents = []

        rng = np.random.RandomState(42)  # 固定种子以保证可重复性
        for _ in range(n_bootstraps):
            idx = rng.choice(n, size=n, replace=True)
            y_boot = y_true[idx]
            old_boot = proba_old[idx]
            new_boot = proba_new[idx]

            pred_old_boot = (old_boot >= th).astype(int)
            pred_new_boot = (new_boot >= th).astype(int)

            # 病例
            case_boot = (y_boot == 1)
            n_cases_boot = np.sum(case_boot)
            if n_cases_boot > 0:
                up_cases_b = np.sum((pred_old_boot[case_boot] == 0) & (pred_new_boot[case_boot] == 1))
                down_cases_b = np.sum((pred_old_boot[case_boot] == 1) & (pred_new_boot[case_boot] == 0))
                nri_events_b = (up_cases_b - down_cases_b) / n_cases_boot
            else:
                nri_events_b = np.nan

            # 对照
            control_boot = (y_boot == 0)
            n_controls_boot = np.sum(control_boot)
            if n_controls_boot > 0:
                up_controls_b = np.sum((pred_old_boot[control_boot] == 0) & (pred_new_boot[control_boot] == 1))
                down_controls_b = np.sum((pred_old_boot[control_boot] == 1) & (pred_new_boot[control_boot] == 0))
                nri_nonevents_b = (down_controls_b - up_controls_b) / n_controls_boot
            else:
                nri_nonevents_b = np.nan

            nri_b = nri_events_b + nri_nonevents_b if not np.isnan(nri_events_b) and not np.isnan(nri_nonevents_b) else np.nan

            boot_nri.append(nri_b)
            boot_nri_events.append(nri_events_b)
            boot_nri_nonevents.append(nri_nonevents_b)

        # 去除 NaN 值（如果某些重采样子组无病例/对照）
        boot_nri = np.array(boot_nri)[~np.isnan(boot_nri)]
        boot_nri_events = np.array(boot_nri_events)[~np.isnan(boot_nri_events)]
        boot_nri_nonevents = np.array(boot_nri_nonevents)[~np.isnan(boot_nri_nonevents)]

        # 计算百分位数置信区间
        lower = alpha / 2
        upper = 1 - alpha / 2
        ci_nri = (np.percentile(boot_nri, lower*100), np.percentile(boot_nri, upper*100)) if len(boot_nri) > 0 else (np.nan, np.nan)
        ci_events = (np.percentile(boot_nri_events, lower*100), np.percentile(boot_nri_events, upper*100)) if len(boot_nri_events) > 0 else (np.nan, np.nan)
        ci_nonevents = (np.percentile(boot_nri_nonevents, lower*100), np.percentile(boot_nri_nonevents, upper*100)) if len(boot_nri_nonevents) > 0 else (np.nan, np.nan)

        # ---- 3. 汇总结果 ----
        results[th] = {
            'threshold': th,
            'nri': nri,
            'nri_events': nri_events,
            'nri_nonevents': nri_nonevents,
            'up_cases': up_cases,
            'down_cases': down_cases,
            'up_controls': up_controls,
            'down_controls': down_controls,
            'stay_low_cases': stay_low_cases,
            'stay_high_cases': stay_high_cases,
            'stay_low_controls': stay_low_controls,
            'stay_high_controls': stay_high_controls,
            'n_cases': n_cases,
            'n_controls': n_controls,
            'nri_ci_lower': ci_nri[0],
            'nri_ci_upper': ci_nri[1],
            'nri_events_ci_lower': ci_events[0],
            'nri_events_ci_upper': ci_events[1],
            'nri_nonevents_ci_lower': ci_nonevents[0],
            'nri_nonevents_ci_upper': ci_nonevents[1]
        }

    return results