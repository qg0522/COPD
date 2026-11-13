#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import textwrap
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.tools import add_constant
import itertools
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


def parse_arguments():
    out_prefix = "./pheno_odds_ratio"
    co_vars = "age,sex"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='calculate odds ratios using logistic regression',
                                     epilog=textwrap.dedent('''\
                                             version:    1.0
                                             date:    2025/09/11
                                             contact:    qiangang@digena.com.cn'''))

    parser.add_argument("-i", dest="input_table", help="the excel table with phenotypes, can be specified multiple "
                                                       "times", action='append', required=True)
    parser.add_argument("-p", dest="phenotype", help="the phenotype name, can be specified multiple times",
                        action='append', required=True)
    parser.add_argument("-c", dest="category", help=f"the phenotype category, comma-separated, "
                                                    f"the first item is treated as reference category,"
                                                    f"can be specified multiple times", action='append', required=True)
    parser.add_argument("--co_vars", dest="co_vars", help=f"the covariants used to do logistic regression, "
                                                          f"comma-separated, default: {co_vars}", default=co_vars)
    parser.add_argument("-o", dest="out_prefix", help=f"the output prefix, default: {out_prefix}", default=out_prefix)

    args = parser.parse_args()

    return args


def read_phenotypes(input_tables):
    df = pd.DataFrame()
    for index, pheno_table in enumerate(input_tables):
        df1 = pd.read_excel(pheno_table, comment='#')
        df1 = df1.rename(columns={'sample_index': 'sample_id',
                                  'group': 'case',
                                  '吸烟状态（never/former/current)': 'smoking_status',
                                  'Smoking_Status': 'smoking_status',
                                  '年龄': 'age',
                                  'Age': 'age',
                                  '性别': 'sex',
                                  'Gender_chn': 'sex',
                                  'Gender': 'gender',
                                  'ID': 'sample_id',
                                  '身高（cm）': 'height_cm',
                                  'Height_cm': 'height_cm',
                                  '体重（kg）': 'weight_kg',
                                  'Weight_kg': 'weight_kg',
                                  })
        if index == 0:
            df = df1
        else:
            df = df.merge(df1, on="sample_id", how='inner').copy()

    # BMI（可选）
    if 'height_cm' in df.columns and 'weight_kg' in df.columns:
        # 空值处理
        median_height = df['height_cm'].median()
        median_weight = df['weight_kg'].median()
        df = df.fillna({'height_cm': median_height, 'weight_kg': median_weight})

        h_m = df['height_cm'] / 100.0
        df['bmi'] = df['weight_kg'] / (h_m * h_m)
    else:
        df['bmi'] = np.nan

    return df


def get_odds_ratio_by_logistic_regression(df, phenotypes, category_list, co_vars, out_prefix):

    # 构造特征组合
    pheno_comb_col = 'pheno_comb'
    ref_category_list = []
    category_list_split = []
    for index, phenotype in enumerate(phenotypes):
        categorys = str(category_list[index]).split(",")
        category_list_split.append(categorys)
        ref_category = categorys[0]
        ref_category_list.append(ref_category)

        if index == 0:
            df[pheno_comb_col] = df[phenotype]
        else:
            df[pheno_comb_col] = df[pheno_comb_col] + ":" + df[phenotype]

    comb_order = [":".join(comb) for comb in list(itertools.product(*category_list_split))]
    df[pheno_comb_col] = pd.Categorical(df[pheno_comb_col], categories=comb_order, ordered=True)

    # logistic回归
    X = pd.get_dummies(df[[pheno_comb_col]], drop_first=False, prefix_sep=":")
    ref_pheno_comb = ":".join(ref_category_list)
    ref_pheno_comb_with_prefix = pheno_comb_col + ":" + ref_pheno_comb
    X = X.drop(columns=[ref_pheno_comb_with_prefix])  # 删除参考组

    covars = []
    for co_var in str(co_vars).split(","):
        if co_var in df.columns and df[co_var].notna().all():
            covars.append(co_var)
    # 性别转为 0/1
    if 'sex' in covars:
        df['sex'] = df['sex'].map({'男': 1, '女': 0})
    if covars:
        X = pd.concat([X, df[covars]], axis=1)

    X = add_constant(X)
    df['case'] = df['case'].map({'case': 1, 'control': 0})
    y = df['case'].astype(int)

    # 检查特征相关性
    correlation_matrix = X.corr().abs()

    # 找出高度相关的特征对（相关系数 > 0.9）
    high_corr = np.where(correlation_matrix > 0.9)
    high_corr = [(correlation_matrix.columns[x], correlation_matrix.columns[y])
                 for x, y in zip(*high_corr) if x != y and x < y]
    print(high_corr)

    # 检查方差为零的列
    zero_variance_cols = X.columns[X.std() == 0]
    print(zero_variance_cols)

    # 计算VIF
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

    # 通常，如果VIF大于10，则认为存在高度共线性
    high_vif = vif_data[vif_data["VIF"] > 10]
    print(high_vif)

    # # 选择需要标准化的数值列（排除分类变量和常数项）
    # numeric_cols = [col for col in X.columns if col != 'const' and X[col].dtype in ['float64', 'int64']]
    #
    # # 创建标准化器并转换数据
    # scaler = StandardScaler()
    # X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    # 布尔变量转 int
    X = X.astype({c: int for c in X.select_dtypes(bool).columns})
    # 最后转 float
    X = X.astype(float)

    # 拟合 logistic 回归
    logit = sm.Logit(y, X).fit(disp=False)  # default method=newton
    # logit = sm.Logit(y, X).fit(disp=False, method='lbfgs')
    # logit = sm.Logit(y, X).fit(disp=False, method='bfgs', maxiter=100)
    # logit = sm.Logit(y, X).fit(disp=False, method='nm')
    # logit = sm.Logit(y, X).fit(disp=False, method='ng')  # error
    # logit = sm.Logit(y, X).fit(disp=False, method='ncg')  # pass, weird results
    # logit = sm.Logit(y, X).fit(disp=False, method='powell', maxiter=1000)  # weird results
    # logit = sm.Logit(y, X).fit(disp=False, method='basinhopping')
    # logit = sm.Logit(y, X).fit(disp=False, method='minimize')
    # logit = sm.Logit(y, X).fit_regularized(alpha=0.002, disp=False, maxiter=1000) # weird results

    # 导出 OR 表
    params = logit.params
    conf = logit.conf_int()
    case_num = df[(df[pheno_comb_col] == ref_pheno_comb) & (df['case'] == 1)].shape[0]
    control_num = df[(df[pheno_comb_col] == ref_pheno_comb) & (df['case'] == 0)].shape[0]
    rows = [ref_category_list + [case_num, control_num, 1.0, '', '', '']]
    for c in X.columns:
        if c.startswith(pheno_comb_col):
            pheno_comb_list = c.split(':')[1:]
            pheno_comb = ":".join(pheno_comb_list)
            case_num = df[(df[pheno_comb_col] == pheno_comb) & (df['case'] == 1)].shape[0]
            control_num = df[(df[pheno_comb_col] == pheno_comb) & (df['case'] == 0)].shape[0]
            OR = np.exp(params[c])
            L, U = np.exp(conf.loc[c])
            pval = logit.pvalues[c]
            rows.append(pheno_comb_list + [case_num, control_num, OR, L, U, pval])

    or_df = pd.DataFrame(rows, columns=phenotypes + ['case_num', 'control_num', 'OR', 'CI_low', 'CI_high', 'p'])
    or_df.to_csv(f"{out_prefix}_OR.csv", index=False)

    # 保存回归模型摘要
    with open(f"{out_prefix}_logistic_summary.txt", 'w') as f:
        f.write(logit.summary2().as_text())


def main():
    # parse arguments
    args = parse_arguments()

    # read input tables of phenotypes and merge
    df = read_phenotypes(args.input_table)

    # odds ratio by logistic regression
    get_odds_ratio_by_logistic_regression(df, args.phenotype, args.category, args.co_vars, args.out_prefix)


if __name__ == '__main__':
    main()
