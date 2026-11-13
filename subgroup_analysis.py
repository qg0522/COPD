# subgroup_analysis.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from model_config import (ALL_MODELS_CONFIG, FEATURE_COMBINATIONS,
                          SUBGROUP_CONDITIONS, STRAT_COLS, TARGET_COL,
                          VAL_FLAG_COL, SAMPLE_ID_COL)
from model_utils import (load_data, preprocess_data, calculate_metrics, plot_roc_curves, calculate_nri)
import joblib
import numpy as np


def evaluate_subgroup(y_true, y_proba, strat_df, condition, threshold):
    """评估子组性能，condition: dict {col: [values]}"""
    mask = pd.Series(True, index=strat_df.index)
    for col, vals in condition.items():
        orig_col = f"{col}_orig"
        if orig_col in strat_df.columns:
            mask = mask & strat_df[orig_col].isin(vals)
        else:
            return None
    if mask.sum() == 0:
        return None
    y_sub = y_true[mask]
    proba_sub = y_proba[mask]
    return calculate_metrics(y_sub, proba_sub, threshold)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--sheet', default=None)
    parser.add_argument('--model', default='RF', help='使用的模型名称')
    parser.add_argument('--output', default='subgroup_output')
    parser.add_argument('--cv', type=int, default=5)
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 加载数据
    df = load_data(args.data, args.sheet)
    train_df = df[df[VAL_FLAG_COL] == "No"].copy()
    test_df = df[df[VAL_FLAG_COL] == "Yes"].copy()
    
    # 2. 对于每个特征组合
    all_results = {}
    for combo_name, feat_cols in FEATURE_COMBINATIONS.items():
        print(f"\n=== 特征组合：{combo_name} ===")
        X_train, y_train, strat_train, _ = preprocess_data(
            train_df, TARGET_COL, feat_cols, STRAT_COLS, SAMPLE_ID_COL
        )
        X_test, y_test, strat_test, _ = preprocess_data(
            test_df, TARGET_COL, feat_cols, STRAT_COLS, SAMPLE_ID_COL
        )
        
        config = ALL_MODELS_CONFIG[args.model]
        model = config['model']
        param_grid = config['params']
        
        if config['needs_scaling']:
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
        else:
            X_train_scaled = X_train.values
            X_test_scaled = X_test.values
        
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_scaled, y_train)
        best_model = grid.best_estimator_
        best_params = str(grid.best_params_)  # 转为字符串便于存储
        
        y_proba_train = best_model.predict_proba(X_train_scaled)[:, 1]
        train_metrics = calculate_metrics(y_train, y_proba_train)
        threshold = train_metrics['threshold']
        
        y_proba_test = best_model.predict_proba(X_test_scaled)[:, 1]
        test_metrics = calculate_metrics(y_test, y_proba_test, threshold)
        
        # 训练集子组
        train_subgroup_results = {}
        for sub_name, condition in SUBGROUP_CONDITIONS.items():
            metrics = evaluate_subgroup(y_train, y_proba_train, strat_train, condition, threshold)
            if metrics:
                train_subgroup_results[sub_name] = metrics
                print(f"  训练集子组 {sub_name}: AUC={metrics['auc']:.4f}")
        
        # 测试集子组
        test_subgroup_results = {}
        for sub_name, condition in SUBGROUP_CONDITIONS.items():
            metrics = evaluate_subgroup(y_test, y_proba_test, strat_test, condition, threshold)
            if metrics:
                test_subgroup_results[sub_name] = metrics
                print(f"  测试集子组 {sub_name}: AUC={metrics['auc']:.4f}")
        
        all_results[combo_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'train_subgroups': train_subgroup_results,
            'test_subgroups': test_subgroup_results,
            'model': best_model,
            'best_params': best_params,
            'y_train': y_train,
            'y_test': y_test,
            'y_proba_train': y_proba_train,
            'y_proba_test': y_proba_test,
        }
    
    # 3. 保存结果表格（包含训练集和测试集整体 + 子组）
    rows = []
    for combo, res in all_results.items():
        # 训练集整体
        m = res['train']
        rows.append({'combo': combo, 'dataset': 'train', 'subgroup': 'Overall',
                     'auc': m['auc'], 'auc_ci_lower': m['auc_ci_lower'], 'auc_ci_upper': m['auc_ci_upper'],
                     'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'], 'specificity': m['specificity'],
                     'ppv': m['ppv'], 'npv': m['npv'], 'f1': m['f1_score'], 'brier': m['brier'],
                     'threshold': m['threshold'], 'best_params': res['best_params']})
        # 测试集整体
        m = res['test']
        rows.append({'combo': combo, 'dataset': 'test', 'subgroup': 'Overall',
                     'auc': m['auc'], 'auc_ci_lower': m['auc_ci_lower'], 'auc_ci_upper': m['auc_ci_upper'],
                     'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'], 'specificity': m['specificity'],
                     'ppv': m['ppv'], 'npv': m['npv'], 'f1': m['f1_score'], 'brier': m['brier'],
                     'threshold': m['threshold'], 'best_params': res['best_params']})
        # 训练集子组
        for sub, m in res['train_subgroups'].items():
            rows.append({'combo': combo, 'dataset': 'train', 'subgroup': sub,
                         'auc': m['auc'], 'auc_ci_lower': m['auc_ci_lower'], 'auc_ci_upper': m['auc_ci_upper'],
                         'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'], 'specificity': m['specificity'],
                         'ppv': m['ppv'], 'npv': m['npv'], 'f1': m['f1_score'], 'brier': m['brier'],
                         'threshold': m['threshold'], 'best_params': res['best_params']})
        # 测试集子组
        for sub, m in res['test_subgroups'].items():
            rows.append({'combo': combo, 'dataset': 'test', 'subgroup': sub,
                         'auc': m['auc'], 'auc_ci_lower': m['auc_ci_lower'], 'auc_ci_upper': m['auc_ci_upper'],
                         'accuracy': m['accuracy'], 'sensitivity': m['sensitivity'], 'specificity': m['specificity'],
                         'ppv': m['ppv'], 'npv': m['npv'], 'f1': m['f1_score'], 'brier': m['brier'],
                         'threshold': m['threshold'], 'best_params': res['best_params']})
    
    df_out = pd.DataFrame(rows)
    df_out.to_excel(os.path.join(args.output, 'subgroup_metrics.xlsx'), index=False)
    
    # 4. 绘制整体 ROC 曲线（训练集和测试集）
    train_roc_dict = {combo: res['train'] for combo, res in all_results.items()}
    test_roc_dict = {combo: res['test'] for combo, res in all_results.items()}
    plot_roc_curves(train_roc_dict,
                    f'ROC Curves - Training Set',
                    os.path.join(args.output, 'roc_train.png'))
    plot_roc_curves(test_roc_dict,
                    f'ROC Curves - Test Set',
                    os.path.join(args.output, 'roc_test.png'))
    
    # ========== 新增：绘制每个子组的 ROC 曲线 ==========
    # 遍历所有子组
    for sub_name in SUBGROUP_CONDITIONS.keys():
        # 训练集子组 ROC
        train_sub_roc = {}
        for combo, res in all_results.items():
            if sub_name in res['train_subgroups']:
                train_sub_roc[combo] = res['train_subgroups'][sub_name]
        if train_sub_roc:
            plot_roc_curves(train_sub_roc,
                            f'ROC Curves (Train) - Subgroup {sub_name}',
                            os.path.join(args.output, f'roc_train_subgroup_{sub_name}.png'))
        
        # 测试集子组 ROC
        test_sub_roc = {}
        for combo, res in all_results.items():
            if sub_name in res['test_subgroups']:
                test_sub_roc[combo] = res['test_subgroups'][sub_name]
        if test_sub_roc:
            plot_roc_curves(test_sub_roc,
                            f'ROC Curves (Test) - Subgroup {sub_name}',
                            os.path.join(args.output, f'roc_test_subgroup_{sub_name}.png'))

        # ========== 新增：计算 PRS 的 NRI ==========
        # base_combo = 'Age + Gender + BMI + Smoking_Status'
        # ext_combo = 'Age + Gender + BMI + Smoking_Status + PRS'
        base_combo = 'Clinical Feature Model'
        ext_combo = 'Integrated Risk Assessment Model'

        if base_combo in all_results and ext_combo in all_results:
            print("\n=== 计算 PRS 的 NRI (测试集) ===")
            y_test_base = all_results[base_combo]['y_test']
            proba_base = all_results[base_combo]['y_proba_test']
            proba_ext = all_results[ext_combo]['y_proba_test']

            # 确保 y_test 一致
            if not np.array_equal(y_test_base, all_results[ext_combo]['y_test']):
                print("警告：基准组合和扩展组合的测试集标签不一致，请检查数据划分。")
            else:
                thresholds = [0.05, 0.10, 0.20, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                # nri_results = calculate_nri(y_test_base, proba_base, proba_ext, thresholds)

                # # 转换为 DataFrame 保存
                # nri_rows = []
                # for th, stats in nri_results.items():
                #     nri_rows.append({
                #         'threshold': th,
                #         'nri': stats['nri'],
                #         'nri_events': stats['nri_events'],
                #         'nri_nonevents': stats['nri_nonevents'],
                #         'up_cases': stats['up_cases'],
                #         'down_cases': stats['down_cases'],
                #         'up_controls': stats['up_controls'],
                #         'down_controls': stats['down_controls'],
                #         'n_cases': stats['n_cases'],
                #         'n_controls': stats['n_controls']
                #     })

                # 调用增强后的 calculate_nri（注意多了 n_bootstraps 参数）
                nri_results = calculate_nri(y_test_base, proba_base, proba_ext, thresholds, n_bootstraps=1000)

                # 转换为 DataFrame 保存（扩展列）
                nri_rows = []
                for th, stats in nri_results.items():
                    nri_rows.append({
                        'threshold': th,
                        'nri': stats['nri'],
                        'nri_ci_lower': stats['nri_ci_lower'],
                        'nri_ci_upper': stats['nri_ci_upper'],
                        'nri_events': stats['nri_events'],
                        'nri_events_ci_lower': stats['nri_events_ci_lower'],
                        'nri_events_ci_upper': stats['nri_events_ci_upper'],
                        'nri_nonevents': stats['nri_nonevents'],
                        'nri_nonevents_ci_lower': stats['nri_nonevents_ci_lower'],
                        'nri_nonevents_ci_upper': stats['nri_nonevents_ci_upper'],
                        'up_cases': stats['up_cases'],
                        'down_cases': stats['down_cases'],
                        'stay_low_cases': stats['stay_low_cases'],
                        'stay_high_cases': stats['stay_high_cases'],
                        'up_controls': stats['up_controls'],
                        'down_controls': stats['down_controls'],
                        'stay_low_controls': stats['stay_low_controls'],
                        'stay_high_controls': stats['stay_high_controls'],
                        'n_cases': stats['n_cases'],
                        'n_controls': stats['n_controls']
                    })

                df_nri = pd.DataFrame(nri_rows)
                df_nri.to_excel(os.path.join(args.output, 'nri_results.xlsx'), index=False)
                print("NRI 结果已保存至 nri_results.xlsx")
                print(df_nri.to_string(index=False))
        else:
            print(f"警告：未找到基准组合 '{base_combo}' 或扩展组合 '{ext_combo}'，跳过 NRI 计算。")

        # 5. 保存训练好的模型
        models_to_save = {combo: res['model'] for combo, res in all_results.items()}
        joblib.dump(models_to_save, os.path.join(args.output, 'trained_models.pkl'))
        print(f"所有模型已保存至 {os.path.join(args.output, 'trained_models.pkl')}")

        print("完成！结果保存在:", args.output)
    
    print("完成！结果保存在:", args.output)


if __name__ == '__main__':
    main()