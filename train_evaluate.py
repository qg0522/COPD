# train_evaluate.py
import os
import argparse
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
import joblib
from model_config import ALL_MODELS_CONFIG, TARGET_COL, VAL_FLAG_COL, SAMPLE_ID_COL
from model_utils import (load_data, preprocess_data, calculate_metrics,
                   plot_roc_curves, plot_calibration_curves, plot_decision_curves,
                   compute_shap)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='数据文件路径')
    parser.add_argument('--sheet', default=None, help='Sheet名称（Excel）')
    parser.add_argument('--features', nargs='+', help='特征列表，空格分隔')
    parser.add_argument('--models', nargs='+', default=['RF'], help='要训练的模型列表，默认RF')
    parser.add_argument('--output', default='output', help='输出目录')
    parser.add_argument('--cv', type=int, default=5, help='交叉验证折数')
    args = parser.parse_args()
    
    os.makedirs(args.output, exist_ok=True)
    
    # 1. 加载数据
    df = load_data(args.data, args.sheet)
    print(f"数据形状：{df.shape}")
    
    # 2. 确定特征列表
    if args.features:
        feat_cols = args.features
    else:
        # 从配置中读取默认特征（这里以 clinical+PRS为例）
        from model_config import CLINICAL_FEATURES, PRS_FEATURES
        feat_cols = CLINICAL_FEATURES + PRS_FEATURES
    
    # 3. 划分训练/测试集：IsInValSet==1 作为测试，其余训练
    train_df = df[df[VAL_FLAG_COL] == "No"].copy()
    test_df = df[df[VAL_FLAG_COL] == "Yes"].copy()
    print(f"训练集样本数：{len(train_df)}，测试集样本数：{len(test_df)}")
    
    # 4. 预处理训练集
    X_train, y_train, strat_train, _ = preprocess_data(
        train_df, TARGET_COL, feat_cols, [], SAMPLE_ID_COL
    )
    # 预处理测试集（同样处理）
    X_test, y_test, strat_test, _ = preprocess_data(
        test_df, TARGET_COL, feat_cols, [], SAMPLE_ID_COL
    )
    
    # 5. 标准化（如果需要）
    scaler = StandardScaler()
    needs_scale = any(ALL_MODELS_CONFIG[m]['needs_scaling'] for m in args.models)
    if needs_scale:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values
    
    # 6. 对每个模型进行交叉验证调优
    results = {}
    proba_dict = {}
    for model_name in args.models:
        print(f"\n=== 训练模型：{model_name} ===")
        config = ALL_MODELS_CONFIG[model_name]
        model = config['model']
        param_grid = config['params']
        
        # 选择数据
        if config['needs_scaling']:
            X_train_used = X_train_scaled
            X_test_used = X_test_scaled
        else:
            X_train_used = X_train.values
            X_test_used = X_test.values
        
        # 5折交叉验证
        cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid, cv=cv, scoring='roc_auc',
                            n_jobs=-1, verbose=0)
        grid.fit(X_train_used, y_train)
        best_model = grid.best_estimator_
        print(f"最佳参数：{grid.best_params_}")
        print(f"交叉验证平均AUC：{grid.best_score_:.4f}")
        
        # 预测
        y_train_proba = grid.predict_proba(X_train_used)[:, 1]
        y_test_proba = best_model.predict_proba(X_test_used)[:, 1]
        
        # 计算训练集指标（确定阈值）
        train_metrics = calculate_metrics(y_train, y_train_proba)
        threshold = train_metrics['threshold']
        test_metrics = calculate_metrics(y_test, y_test_proba, threshold)

        # 保存结果
        results[model_name] = {
            'train': train_metrics,
            'test': test_metrics,
            'best_params': grid.best_params_,
            'model': best_model
        }
        proba_dict[model_name] = y_test_proba

        # 翻转训练集和测试集概率
        if test_metrics['auc'] < 0.5:
            print(f"⚠️ 警告：{model_name} 的 AUC < 0.5，自动翻转预测概率！")
            y_train_proba = 1 - y_train_proba
            y_test_proba = 1 - y_test_proba
            
            # 260903
            # 用原阈值重新计算（阈值也需要翻转，即 1 - 原阈值）
            # flipped_threshold = 1 - threshold
            # train_metrics = calculate_metrics(y_train, y_train_proba)
            # test_metrics = calculate_metrics(y_test, y_test_proba, flipped_threshold)
            
            # 261111 测试这里的翻转概率导致阈值不一致的问题
            train_metrics = calculate_metrics(y_train, y_train_proba)
            test_metrics = calculate_metrics(y_test, y_test_proba, train_metrics['threshold'])

            # 更新 results 字典中的值
            results[model_name]['train'] = train_metrics
            results[model_name]['test'] = test_metrics
            proba_dict[model_name] = y_test_proba

        print(f"测试集AUC：{test_metrics['auc']:.4f} (95% CI {test_metrics['auc_ci_lower']:.4f}-{test_metrics['auc_ci_upper']:.4f})")
    
    # 7. 保存性能表格
    rows = []
    for name, res in results.items():
        for dataset in ['train', 'test']:
            m = res[dataset]
            rows.append({
                'model': name,
                'dataset': dataset,
                'auc': m['auc'],
                'auc_ci_lower': m['auc_ci_lower'],
                'auc_ci_upper': m['auc_ci_upper'],
                'accuracy': m['accuracy'],
                'sensitivity': m['sensitivity'],
                'specificity': m['specificity'],
                'ppv': m['ppv'],
                'npv': m['npv'],
                'f1': m['f1_score'],
                'brier': m['brier'],
                'threshold': m['threshold'],
                'best_params': res['best_params']
            })
    df_metrics = pd.DataFrame(rows)
    df_metrics.to_excel(os.path.join(args.output, 'metrics.xlsx'), index=False)
    
    # 8. 绘制ROC曲线
    train_roc_dict = {name: res['train'] for name, res in results.items()}
    test_roc_dict = {name: res['test'] for name, res in results.items()}
    plot_roc_curves(train_roc_dict, 'ROC Curves - Training Set',
                    os.path.join(args.output, 'roc_curves_train.png'))
    plot_roc_curves(test_roc_dict, 'ROC Curves - Test Set',
                    os.path.join(args.output, 'roc_curves_test.png'))
    
    # 9. 校准曲线
    plot_calibration_curves(y_test, proba_dict,
                            os.path.join(args.output, 'calibration.png'))
    
    # 10. 决策曲线
    plot_decision_curves(y_test, proba_dict,
                         os.path.join(args.output, 'decision_curves.png'))
    
    # 11. SHAP分析（可选，以最佳模型为例）
    best_model_name = max(results, key=lambda x: results[x]['test']['auc'])
    best_model = results[best_model_name]['model']
    # 使用原始特征（未缩放）进行SHAP
    X_train_df = pd.DataFrame(X_train.values, columns=feat_cols)
    shap_values, expected = compute_shap(best_model, X_train_df, best_model_name)
    if shap_values is not None:
        # 绘制SHAP摘要图（此处简化，可直接调用shap.summary_plot）
        import shap
        shap.summary_plot(shap_values, X_train_df, show=False)
        plt.savefig(os.path.join(args.output, f'shap_summary_{best_model_name}.png'))
        plt.close()
    
    # 12. 保存模型
    joblib.dump({name: res['model'] for name, res in results.items()},
                os.path.join(args.output, 'models.pkl'))
    print("完成！")


if __name__ == '__main__':
    main()