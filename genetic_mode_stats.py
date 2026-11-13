import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import argparse
import sys
import warnings
from statsmodels.stats.multitest import multipletests   # 新增：用于FDR校正

warnings.filterwarnings('ignore')


def read_ped_map_files(ped_file, map_file):
    """读取PLINK格式的.ped和.map文件，并从第六列获取分组信息"""

    # 读取.map文件
    map_data = pd.read_csv(map_file, sep='\t', header=None,
                           names=['chrom', 'snp_id', 'cM', 'pos'])

    # 读取.ped文件
    ped_data = pd.read_csv(ped_file, sep='\t', header=None)

    # 提取样本信息
    n_columns = ped_data.shape[1]
    n_snps = (n_columns - 6) // 2

    # 提取前6列的基本信息
    sample_info = ped_data.iloc[:, :6]
    sample_info.columns = ['FamilyID', 'IndividualID', 'PaternalID',
                           'MaternalID', 'Sex', 'Phenotype']

    # 从第六列获取分组信息：1=对照，2=病例
    sample_info['Case'] = sample_info['Phenotype'].apply(lambda x: 1 if x == 2 else 0)

    # 提取基因型数据
    genotype_data = {}
    snp_ids = map_data['snp_id'].tolist()

    # 确保SNP数量匹配
    if n_snps != len(snp_ids):
        print(f"警告: .ped文件有{n_snps}个SNP，但.map文件有{len(snp_ids)}个SNP")
        n_snps = min(n_snps, len(snp_ids))
        snp_ids = snp_ids[:n_snps]

    # 解析每个SNP的基因型
    for i in range(n_snps):
        start_col = 6 + i * 2
        # 合并两个等位基因
        genotypes = ped_data.iloc[:, start_col].astype(str) + ped_data.iloc[:, start_col + 1].astype(str)
        genotype_data[snp_ids[i]] = genotypes.tolist()

    # 创建DataFrame
    genotype_df = pd.DataFrame(genotype_data)
    genotype_df.index = sample_info['IndividualID']

    # 返回基因型数据、分组信息和map数据
    return genotype_df, sample_info[['IndividualID', 'Case']], map_data


def read_clinical_data(excel_file, sheet_name, sample_id_col, covariate_cols):
    """读取临床信息Excel文件，只读取协变量列"""

    # 读取Excel文件
    df = pd.read_excel(excel_file, sheet_name=sheet_name, comment='#')

    # 检查必需的列是否存在
    required_cols = [sample_id_col] + covariate_cols
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        print(f"错误: 临床数据中缺少必需的列: {missing_cols}")
        print(f"可用列: {list(df.columns)}")
        sys.exit(1)

    # 选择必要的列
    selected_cols = [sample_id_col] + covariate_cols
    clinical_df = df[selected_cols].copy()

    # 处理协变量中的性别列（如果存在）
    for col in covariate_cols:
        if '性别' in col or 'gender' in col.lower() or 'sex' in col.lower():
            # 将性别转换为数值
            clinical_df[col] = clinical_df[col].map({'男': 1, '女': 0, 'M': 1, 'F': 0, 1: 1, 0: 0,
                                                     'male': 1, 'female': 0})

    # 重命名协变量列，移除空格等特殊字符
    for col in covariate_cols:
        new_col = col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
        clinical_df = clinical_df.rename(columns={col: new_col})

    # 更新协变量列名
    covariate_cols_clean = [col.replace(' ', '_').replace('(', '').replace(')', '').replace('-', '_')
                            for col in covariate_cols]

    # 处理缺失值
    clinical_df = clinical_df.dropna(subset=[sample_id_col] + covariate_cols_clean)

    return clinical_df, covariate_cols_clean


def encode_genotypes(genotype_series):
    """编码基因型为数值，同时保留碱基信息"""
    # 统计等位基因频率
    alleles = []
    for gt in genotype_series:
        if pd.isna(gt) or gt == '00' or gt == '--' or gt == '0' or len(str(gt)) < 2:
            continue
        alleles.extend([gt[0], gt[1]])

    if not alleles:
        return None, None, None, None

    # 找到次要等位基因
    from collections import Counter
    allele_counts = Counter(alleles)
    # 排除'0'等缺失值标记
    if '0' in allele_counts:
        del allele_counts['0']

    if not allele_counts:
        return None, None, None, None

    major_allele, major_count = allele_counts.most_common(1)[0]

    # 确定次要等位基因
    minor_allele = None
    for allele, count in allele_counts.items():
        if allele != major_allele:
            minor_allele = allele
            break

    if minor_allele is None:
        # 所有等位基因相同
        minor_allele = '0'

    # 编码基因型
    encoded = []
    genotype_strings = []
    for gt in genotype_series:
        if pd.isna(gt) or gt == '00' or gt == '--' or gt == '0' or len(str(gt)) < 2:
            encoded.append(np.nan)
            genotype_strings.append('')
        else:
            # 计算次要等位基因数量
            minor_count = gt.count(minor_allele)
            encoded.append(minor_count)
            # 生成碱基形式的基因型字符串
            if minor_count == 0:
                genotype_strings.append(f"{major_allele}|{major_allele}")
            elif minor_count == 2:
                genotype_strings.append(f"{minor_allele}|{minor_allele}")
            else:  # minor_count == 1
                # 杂合子，按字母顺序排序
                alleles_sorted = sorted([major_allele, minor_allele])
                genotype_strings.append(f"{alleles_sorted[0]}|{alleles_sorted[1]}")

    return pd.Series(encoded, index=genotype_series.index), pd.Series(genotype_strings,
                                                                      index=genotype_series.index), major_allele, minor_allele


def get_genotype_counts(data, snp_col, genotype_str_col):
    """统计基因型计数"""
    counts = {}
    genotype_counts = {}

    # 统计每个编码值对应的计数
    for idx, row in data.iterrows():
        genotype_code = row[snp_col]
        genotype_str = row[genotype_str_col]

        if pd.isna(genotype_code) or genotype_str == '':
            continue

        if genotype_code not in genotype_counts:
            genotype_counts[genotype_code] = {'count': 0, 'genotype_str': genotype_str}

        genotype_counts[genotype_code]['count'] += 1

    return genotype_counts


def get_genotype_str_for_code(genotype_counts, code, major_allele, minor_allele):
    """根据编码获取基因型字符串"""
    if code in genotype_counts and genotype_counts[code]['genotype_str']:
        return genotype_counts[code]['genotype_str']
    else:
        # 根据等位基因生成
        if code == 0:
            return f"{major_allele}|{major_allele}"
        elif code == 2:
            return f"{minor_allele}|{minor_allele}"
        else:  # code == 1
            alleles_sorted = sorted([major_allele, minor_allele])
            return f"{alleles_sorted[0]}|{alleles_sorted[1]}"


def build_genotype_str_map(control_genotype_counts, case_genotype_counts, major_allele, minor_allele):
    """构建编码到基因型字符串的映射"""
    genotype_str_map = {}

    for code in [0, 1, 2]:
        # 优先使用对照组中的基因型字符串，如果没有则使用病例组的，如果都没有则根据等位基因生成
        control_str = get_genotype_str_for_code(control_genotype_counts, code, major_allele, minor_allele)
        case_str = get_genotype_str_for_code(case_genotype_counts, code, major_allele, minor_allele)

        # 如果两个字符串不同，选择非空的
        if control_str and case_str:
            genotype_str_map[code] = control_str  # 优先使用对照组的
        elif control_str:
            genotype_str_map[code] = control_str
        elif case_str:
            genotype_str_map[code] = case_str
        else:
            # 根据等位基因生成
            if code == 0:
                genotype_str_map[code] = f"{major_allele}|{major_allele}"
            elif code == 2:
                genotype_str_map[code] = f"{minor_allele}|{minor_allele}"
            else:  # code == 1
                alleles_sorted = sorted([major_allele, minor_allele])
                genotype_str_map[code] = f"{alleles_sorted[0]}|{alleles_sorted[1]}"

    return genotype_str_map


def perform_logistic_regression(data, snp_col, model_type, covariates):
    """执行逻辑回归分析"""

    # 准备数据，删除缺失值
    analysis_data = data[['Case', snp_col] + covariates].dropna()

    if len(analysis_data) < 10:  # 样本量太小
        return None, None, None, None

    # 根据模型类型处理SNP变量
    if model_type == 'allelic':
        # 等位基因模型：每个次要等位基因增加的风险
        formula = f'Case ~ {snp_col} + ' + ' + '.join(covariates)

    elif model_type == 'dominant':
        # 显性模型：携带至少一个次要等位基因 vs 不携带
        analysis_data[f'{snp_col}_dom'] = analysis_data[snp_col].apply(lambda x: 1 if x >= 1 else 0)
        formula = f'Case ~ {snp_col}_dom + ' + ' + '.join(covariates)

    elif model_type == 'recessive':
        # 隐性模型：两个次要等位基因 vs 其他
        analysis_data[f'{snp_col}_rec'] = analysis_data[snp_col].apply(lambda x: 1 if x == 2 else 0)
        formula = f'Case ~ {snp_col}_rec + ' + ' + '.join(covariates)

    elif model_type == 'codominant':
        # 共显性模型：使用分类变量
        analysis_data[f'{snp_col}_cat'] = pd.Categorical(analysis_data[snp_col])
        formula = f'Case ~ C({snp_col}_cat) + ' + ' + '.join(covariates)

    elif model_type == 'overdominant':
        # 超显性模型：杂合子 vs 纯合子
        analysis_data[f'{snp_col}_over'] = analysis_data[snp_col].apply(lambda x: 1 if x == 1 else 0)
        formula = f'Case ~ {snp_col}_over + ' + ' + '.join(covariates)

    else:
        raise ValueError(f"未知的模型类型: {model_type}")

    try:
        # 执行逻辑回归
        model = smf.logit(formula, data=analysis_data).fit(disp=False, maxiter=100)

        # 提取结果
        params = model.params
        conf_int = model.conf_int()
        p_values = model.pvalues

        # 计算OR和95% CI
        or_values = np.exp(params)
        ci_lower = np.exp(conf_int[0])
        ci_upper = np.exp(conf_int[1])

        return or_values, ci_lower, ci_upper, p_values

    except Exception as e:
        print(f"回归分析失败: {e}")
        return None, None, None, None


def extract_or_p_values(or_values, ci_lower, ci_upper, p_values, index):
    """从回归结果中提取指定索引的OR值、置信区间和P值"""
    if or_values is None or len(or_values) <= index:
        return np.nan, np.nan, np.nan, np.nan

    or_val = or_values.iloc[index] if index < len(or_values) else np.nan
    ci_low = ci_lower.iloc[index] if index < len(ci_lower) else np.nan
    ci_up = ci_upper.iloc[index] if index < len(ci_upper) else np.nan
    p_val = p_values.iloc[index] if index < len(p_values) else np.nan

    return or_val, ci_low, ci_up, p_val


def format_or_value(or_val, ci_low, ci_up, threshold=1e6):
    """格式化OR值，特别大的值显示为Inf"""
    if np.isnan(or_val) or np.isnan(ci_low) or np.isnan(ci_up):
        return 'NA', 'NA'

    # print("or_val, ci_low, ci_up", or_val, ci_low, ci_up)

    # 设置阈值，当OR值超过阈值时显示为Inf
    if or_val >= threshold:
        or_str = "Inf"
    else:
        or_str = f"{or_val:.2f}"

    # 格式化置信区间
    if ci_low >= threshold and ci_up >= threshold:
        ci_str = "Inf-Inf"
    elif ci_low >= threshold:
        ci_str = f"Inf-{ci_up:.2f}"
    elif ci_up >= threshold:
        ci_str = f"{ci_low:.2f}-Inf"
    else:
        ci_str = f"{ci_low:.2f}-{ci_up:.2f}"

    return or_str, ci_str


def format_results(snp_id, control_genotype_counts, case_genotype_counts,
                   or_values, ci_lower, ci_upper, p_values, model_type,
                   major_allele, minor_allele, control_total, case_total):
    """格式化结果，显示碱基形式的基因型和频率"""

    results = []

    # 构建编码到基因型字符串的映射
    genotype_str_map = build_genotype_str_map(control_genotype_counts, case_genotype_counts, major_allele, minor_allele)

    if model_type == 'codominant':
        # 共显性模型有多个比较
        for genotype_code in [0, 1, 2]:
            control_count = control_genotype_counts.get(genotype_code, {'count': 0, 'genotype_str': ''})
            case_count = case_genotype_counts.get(genotype_code, {'count': 0, 'genotype_str': ''})

            # 获取基因型字符串
            genotype_str = genotype_str_map[genotype_code]

            control_percent = (control_count['count'] / control_total * 100) if control_total > 0 else 0
            case_percent = (case_count['count'] / case_total * 100) if case_total > 0 else 0

            if genotype_code == 0:
                # 参考基因型
                results.append({
                    'SNP位点': snp_id,
                    '遗传模型': '共显性',
                    '基因型/等位基因': genotype_str,
                    '对照组(%)': f"{control_count['count']} ({control_percent:.1f}%)",
                    '病例组(%)': f"{case_count['count']} ({case_percent:.1f}%)",
                    'Adjusted OR(95% CI)': '1.00',
                    'P': ''
                })
            else:
                # 非参考基因型
                or_val, ci_low, ci_up, p_val = extract_or_p_values(or_values, ci_lower, ci_upper, p_values,
                                                                   genotype_code)

                # 格式化OR值和置信区间
                or_str, ci_str = format_or_value(or_val, ci_low, ci_up)

                results.append({
                    'SNP位点': snp_id,
                    '遗传模型': '共显性',
                    '基因型/等位基因': genotype_str,
                    '对照组(%)': f"{control_count['count']} ({control_percent:.1f}%)",
                    '病例组(%)': f"{case_count['count']} ({case_percent:.1f}%)",
                    'Adjusted OR(95% CI)': f"{or_str}({ci_str})" if or_str != 'NA' else 'NA',
                    'P': f"{p_val:.4f}" if not np.isnan(p_val) else 'NA'
                })

    elif model_type == 'allelic':
        # 等位基因模型：计算等位基因频率
        # 计算主要等位基因频率
        control_major_alleles = 0
        control_total_alleles = 0
        for genotype_code, info in control_genotype_counts.items():
            control_major_alleles += (2 - genotype_code) * info['count']  # 每个个体有2个等位基因，减去次要等位基因数得到主要等位基因数
            control_total_alleles += 2 * info['count']

        case_major_alleles = 0
        case_total_alleles = 0
        for genotype_code, info in case_genotype_counts.items():
            case_major_alleles += (2 - genotype_code) * info['count']
            case_total_alleles += 2 * info['count']

        # 计算次要等位基因频率
        control_minor_alleles = control_total_alleles - control_major_alleles
        case_minor_alleles = case_total_alleles - case_major_alleles

        control_major_percent = (
                control_major_alleles / control_total_alleles * 100) if control_total_alleles > 0 else 0
        case_major_percent = (case_major_alleles / case_total_alleles * 100) if case_total_alleles > 0 else 0
        control_minor_percent = (
                control_minor_alleles / control_total_alleles * 100) if control_total_alleles > 0 else 0
        case_minor_percent = (case_minor_alleles / case_total_alleles * 100) if case_total_alleles > 0 else 0

        # 获取OR值和P值
        or_val, ci_low, ci_up, p_val = extract_or_p_values(or_values, ci_lower, ci_upper, p_values, 1)

        # 格式化OR值和置信区间
        or_str, ci_str = format_or_value(or_val, ci_low, ci_up)

        # 第一行：参考等位基因（主要等位基因）
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '等位基因',
            '基因型/等位基因': major_allele,
            '对照组(%)': f"{control_major_alleles} ({control_major_percent:.1f}%)",
            '病例组(%)': f"{case_major_alleles} ({case_major_percent:.1f}%)",
            'Adjusted OR(95% CI)': '1.00',
            'P': ''
        })

        # 第二行：比较等位基因（次要等位基因）
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '等位基因',
            '基因型/等位基因': minor_allele,
            '对照组(%)': f"{control_minor_alleles} ({control_minor_percent:.1f}%)",
            '病例组(%)': f"{case_minor_alleles} ({case_minor_percent:.1f}%)",
            'Adjusted OR(95% CI)': f"{or_str}({ci_str})" if or_str != 'NA' else 'NA',
            'P': f"{p_val:.4f}" if not np.isnan(p_val) else 'NA'
        })

    elif model_type == 'dominant':
        # 显性模型：0 vs (1+2)
        # 第一组：纯合子major (0)
        control_group1 = control_genotype_counts.get(0, {'count': 0, 'genotype_str': genotype_str_map[0]})
        case_group1 = case_genotype_counts.get(0, {'count': 0, 'genotype_str': genotype_str_map[0]})

        # 第二组：至少有一个minor等位基因 (1+2)
        control_group2_count = sum(info['count'] for code, info in control_genotype_counts.items() if code >= 1)
        case_group2_count = sum(info['count'] for code, info in case_genotype_counts.items() if code >= 1)

        # 获取第二组的基因型字符串表示：基因型1和基因型2相加
        genotype_str_1 = genotype_str_map[1]
        genotype_str_2 = genotype_str_map[2]
        group2_genotype_str = f"{genotype_str_1}+{genotype_str_2}"

        control_percent1 = (control_group1['count'] / control_total * 100) if control_total > 0 else 0
        case_percent1 = (case_group1['count'] / case_total * 100) if case_total > 0 else 0
        control_percent2 = (control_group2_count / control_total * 100) if control_total > 0 else 0
        case_percent2 = (case_group2_count / case_total * 100) if case_total > 0 else 0

        # 获取OR值和P值
        or_val, ci_low, ci_up, p_val = extract_or_p_values(or_values, ci_lower, ci_upper, p_values, 1)

        # 格式化OR值和置信区间
        or_str, ci_str = format_or_value(or_val, ci_low, ci_up)

        # 第一行：参考组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '显性',
            '基因型/等位基因': control_group1['genotype_str'],
            '对照组(%)': f"{control_group1['count']} ({control_percent1:.1f}%)",
            '病例组(%)': f"{case_group1['count']} ({case_percent1:.1f}%)",
            'Adjusted OR(95% CI)': '1.00',
            'P': ''
        })

        # 第二行：比较组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '显性',
            '基因型/等位基因': group2_genotype_str,
            '对照组(%)': f"{control_group2_count} ({control_percent2:.1f}%)",
            '病例组(%)': f"{case_group2_count} ({case_percent2:.1f}%)",
            'Adjusted OR(95% CI)': f"{or_str}({ci_str})" if or_str != 'NA' else 'NA',
            'P': f"{p_val:.4f}" if not np.isnan(p_val) else 'NA'
        })

    elif model_type == 'recessive':
        # 隐性模型：(0+1) vs 2
        # 第一组：0+1 (不包含两个minor等位基因)
        control_group1_count = sum(info['count'] for code, info in control_genotype_counts.items() if code <= 1)
        case_group1_count = sum(info['count'] for code, info in case_genotype_counts.items() if code <= 1)

        # 第二组：纯合子minor (2)
        control_group2 = control_genotype_counts.get(2, {'count': 0, 'genotype_str': genotype_str_map[2]})
        case_group2 = case_genotype_counts.get(2, {'count': 0, 'genotype_str': genotype_str_map[2]})

        # 获取第一组的基因型字符串表示：基因型0和基因型1相加
        genotype_str_0 = genotype_str_map[0]
        genotype_str_1 = genotype_str_map[1]
        group1_genotype_str = f"{genotype_str_0}+{genotype_str_1}"

        control_percent1 = (control_group1_count / control_total * 100) if control_total > 0 else 0
        case_percent1 = (case_group1_count / case_total * 100) if case_total > 0 else 0
        control_percent2 = (control_group2['count'] / control_total * 100) if control_total > 0 else 0
        case_percent2 = (case_group2['count'] / case_total * 100) if case_total > 0 else 0

        # 获取OR值和P值
        or_val, ci_low, ci_up, p_val = extract_or_p_values(or_values, ci_lower, ci_upper, p_values, 1)

        # 格式化OR值和置信区间
        or_str, ci_str = format_or_value(or_val, ci_low, ci_up)

        # 第一行：参考组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '隐性',
            '基因型/等位基因': group1_genotype_str,
            '对照组(%)': f"{control_group1_count} ({control_percent1:.1f}%)",
            '病例组(%)': f"{case_group1_count} ({case_percent1:.1f}%)",
            'Adjusted OR(95% CI)': '1.00',
            'P': ''
        })

        # 第二行：比较组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '隐性',
            '基因型/等位基因': control_group2['genotype_str'],
            '对照组(%)': f"{control_group2['count']} ({control_percent2:.1f}%)",
            '病例组(%)': f"{case_group2['count']} ({case_percent2:.1f}%)",
            'Adjusted OR(95% CI)': f"{or_str}({ci_str})" if or_str != 'NA' else 'NA',
            'P': f"{p_val:.4f}" if not np.isnan(p_val) else 'NA'
        })

    elif model_type == 'overdominant':
        # 超显性模型：(0+2) vs 1
        # 第一组：纯合子 (0+2)
        control_group1_count = sum(info['count'] for code, info in control_genotype_counts.items() if code != 1)
        case_group1_count = sum(info['count'] for code, info in case_genotype_counts.items() if code != 1)

        # 第二组：杂合子 (1)
        control_group2 = control_genotype_counts.get(1, {'count': 0, 'genotype_str': genotype_str_map[1]})
        case_group2 = case_genotype_counts.get(1, {'count': 0, 'genotype_str': genotype_str_map[1]})

        # 获取第一组的基因型字符串表示：基因型0和基因型2相加
        genotype_str_0 = genotype_str_map[0]
        genotype_str_2 = genotype_str_map[2]
        group1_genotype_str = f"{genotype_str_0}+{genotype_str_2}"

        control_percent1 = (control_group1_count / control_total * 100) if control_total > 0 else 0
        case_percent1 = (case_group1_count / case_total * 100) if case_total > 0 else 0
        control_percent2 = (control_group2['count'] / control_total * 100) if control_total > 0 else 0
        case_percent2 = (case_group2['count'] / case_total * 100) if case_total > 0 else 0

        # 获取OR值和P值
        or_val, ci_low, ci_up, p_val = extract_or_p_values(or_values, ci_lower, ci_upper, p_values, 1)

        # 格式化OR值和置信区间
        or_str, ci_str = format_or_value(or_val, ci_low, ci_up)

        # 第一行：参考组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '超显性',
            '基因型/等位基因': group1_genotype_str,
            '对照组(%)': f"{control_group1_count} ({control_percent1:.1f}%)",
            '病例组(%)': f"{case_group1_count} ({case_percent1:.1f}%)",
            'Adjusted OR(95% CI)': '1.00',
            'P': ''
        })

        # 第二行：比较组
        results.append({
            'SNP位点': snp_id,
            '遗传模型': '超显性',
            '基因型/等位基因': control_group2['genotype_str'],
            '对照组(%)': f"{control_group2['count']} ({control_percent2:.1f}%)",
            '病例组(%)': f"{case_group2['count']} ({case_percent2:.1f}%)",
            'Adjusted OR(95% CI)': f"{or_str}({ci_str})" if or_str != 'NA' else 'NA',
            'P': f"{p_val:.4f}" if not np.isnan(p_val) else 'NA'
        })

    return results


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='SNP与疾病关联的逻辑回归分析')
    parser.add_argument('--ped', required=True, help='PLINK .ped文件路径')
    parser.add_argument('--map', required=True, help='PLINK .map文件路径')
    parser.add_argument('--clinical', required=True, help='临床信息Excel文件路径')
    parser.add_argument('--sheet_name', required=True, help='sheet名称')
    parser.add_argument('--output', default='snp_analysis_results.csv', help='输出文件路径')
    parser.add_argument('--models', default='codominant,dominant,recessive,overdominant,allelic',
                        help='遗传模型列表，用逗号分隔')
    parser.add_argument('--sample_id_col', default='SampleID',
                        help='临床数据中样本ID列名')
    parser.add_argument('--covariates', required=True,
                        help='校正用到的临床信息列，用逗号分隔，例如: Age,Gender,BMI')

    args = parser.parse_args()

    # 解析协变量列
    covariate_cols = [col.strip() for col in args.covariates.split(',')]

    # 读取数据
    print("读取基因型数据和分组信息...")
    genotype_df, phenotype_info, map_data = read_ped_map_files(args.ped, args.map)

    print("读取临床数据...")
    clinical_df, covariate_cols_clean = read_clinical_data(
        args.clinical, args.sheet_name, args.sample_id_col, covariate_cols
    )

    # 合并数据
    print("合并数据...")
    # 先将分组信息合并到基因型数据
    merged_data = genotype_df.join(phenotype_info.set_index('IndividualID'), how='inner')

    # 再将临床数据合并进来
    merged_data = merged_data.join(clinical_df.set_index(args.sample_id_col), how='inner')

    # 分离病例和对照组
    case_data = merged_data[merged_data['Case'] == 1]
    control_data = merged_data[merged_data['Case'] == 0]

    control_total = len(control_data)
    case_total = len(case_data)

    print(f"样本统计: 总样本数={len(merged_data)}, 病例数={case_total}, 对照数={control_total}")

    # 解析遗传模型
    models = args.models.split(',')

    # 分析每个SNP
    all_results = []

    for snp_id in genotype_df.columns:
        print(f"分析SNP: {snp_id}")

        # 编码基因型，同时获取碱基字符串
        encoded_series, genotype_str_series, major_allele, minor_allele = encode_genotypes(merged_data[snp_id])

        if encoded_series is None:
            print(f"  SNP {snp_id} 无法编码，跳过")
            continue

        # 添加到数据中
        merged_data_encoded = merged_data.copy()
        merged_data_encoded[f'{snp_id}_encoded'] = encoded_series
        merged_data_encoded[f'{snp_id}_genotype_str'] = genotype_str_series

        # 分离病例和对照组（更新后的数据）
        case_data_encoded = merged_data_encoded[merged_data_encoded['Case'] == 1]
        control_data_encoded = merged_data_encoded[merged_data_encoded['Case'] == 0]

        # 统计基因型频率（包括碱基字符串）
        control_genotype_counts = get_genotype_counts(control_data_encoded, f'{snp_id}_encoded',
                                                      f'{snp_id}_genotype_str')
        case_genotype_counts = get_genotype_counts(case_data_encoded, f'{snp_id}_encoded', f'{snp_id}_genotype_str')

        # 对每个遗传模型进行分析
        for model_type in models:
            model_type = model_type.strip()

            print(f"  运行 {model_type} 模型...")

            # 执行逻辑回归
            or_values, ci_lower, ci_upper, p_values = perform_logistic_regression(
                merged_data_encoded, f'{snp_id}_encoded', model_type, covariate_cols_clean
            )

            # 格式化结果
            results = format_results(
                snp_id, control_genotype_counts, case_genotype_counts,
                or_values, ci_lower, ci_upper, p_values, model_type,
                major_allele, minor_allele, control_total, case_total
            )
            all_results.extend(results)

    # 保存结果
    if all_results:
        results_df = pd.DataFrame(all_results)

        # ==================== 新增 FDR 多重校正 ====================
        # 1. 全局 Benjamini-Hochberg 校正（所有比较放在一起）
        # 提取所有有效的 P 值（排除空字符串和参考组的 P 值）
        p_values_for_fdr = []
        p_indices = []
        for idx, val in enumerate(results_df['P']):
            # 如果 P 列是字符串，尝试转换为浮点数；如果是数值直接使用
            if isinstance(val, str):
                if val.replace('.', '', 1).isdigit():  # 简单判断是否为数字字符串（包括小数）
                    p_values_for_fdr.append(float(val))
                    p_indices.append(idx)
            elif isinstance(val, (int, float)):
                if not pd.isna(val):
                    p_values_for_fdr.append(float(val))
                    p_indices.append(idx)

        if p_values_for_fdr:
            # 执行 BH 校正
            _, pvals_corrected, _, _ = multipletests(p_values_for_fdr, alpha=0.05, method='fdr_bh')
            # 创建全局 FDR 列，默认空字符串
            results_df['FDR_BH'] = ''
            for i, orig_idx in enumerate(p_indices):
                results_df.at[orig_idx, 'FDR_BH'] = f"{pvals_corrected[i]:.4f}"
        else:
            results_df['FDR_BH'] = ''

        # 2. 按遗传模型分层 FDR 校正（作为辅助参考）
        results_df['Model-wise_FDR_BH'] = ''
        for model in results_df['遗传模型'].unique():
            # 获取该模型下所有有效的 P 值索引
            model_mask = (results_df['遗传模型'] == model) & (results_df['FDR_BH'] != '')
            model_indices = results_df[model_mask].index.tolist()
            if model_indices:
                model_pvals = [float(results_df.at[i, 'P']) for i in model_indices]
                _, model_corrected, _, _ = multipletests(model_pvals, alpha=0.05, method='fdr_bh')
                for i, idx in enumerate(model_indices):
                    results_df.at[idx, 'Model-wise_FDR_BH'] = f"{model_corrected[i]:.4f}"
        # ========================================================

        results_df.to_csv(args.output, index=False, encoding='utf-8-sig')
        print(f"分析完成！结果已保存到: {args.output}")

        # 打印前几行结果
        print("\n前10行结果:")
        print(results_df.head(10).to_string(index=False))

        # 打印结果统计
        print(f"\n总共分析了 {len(genotype_df.columns)} 个SNP位点")
        print(f"生成了 {len(results_df)} 行结果")
    else:
        print("没有生成任何结果！")


if __name__ == "__main__":
    main()