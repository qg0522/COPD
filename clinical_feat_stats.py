import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, levene, chi2_contingency, fisher_exact
from typing import List, Dict, Optional


def generate_clinical_stats_table(
        data_path: str,
        sheet_name: str = "Sheet1",
        group_col: str = "分组",
        case_label: str = "病例",
        control_label: str = "对照",
        target_vars: List[str] = None,
        var_type_specify: Optional[Dict[str, str]] = None,
        continuous_nunique_thresh: int = 10,
        output_path: Optional[str] = None,
        decimal_places: int = 2,
        p_decimal_places: int = 3
) -> pd.DataFrame:
    """
    生成临床研究基本统计信息表，支持连续型/离散型变量的自动统计与检验。

    参数说明：
    ----------
    data_path: str
        Excel数据文件路径（如"临床信息示例.xlsx"）
    sheet_name: str, default="Sheet1"
        Excel中的工作表名称
    group_col: str, default="分组"
        分组列名（如"分组"列，值为"病例"/"对照"）
    case_label: str, default="病例"
        病例组的分组值（需与Excel中完全一致）
    control_label: str, default="对照"
        对照组的分组值（需与Excel中完全一致）
    target_vars: List[str], optional
        需统计的临床指标列表（如["年龄", "性别", "吸烟状态"]），默认统计所有非分组列
    var_type_specify: Dict[str, str], optional
        手动指定变量类型（优先级高于自动判断），格式：{变量名: "continuous"/"discrete"}
    continuous_nunique_thresh: int, default=10
        自动判断连续型变量的阈值：数值型变量唯一值数>该阈值则视为连续型
    output_path: str, optional
        结果表格输出路径（如"临床统计结果.xlsx"），None则不输出文件
    decimal_places: int, default=1
        均值、标准差的小数保留位数（如1位：72.1±8.5）
    p_decimal_places: int, default=3
        p值的小数保留位数（<0.001时显示"<0.001"）

    返回：
    ----------
    pd.DataFrame: 统计结果表格（可直接打印或保存）
    """
    # 1. 读取数据
    try:
        df = pd.read_excel(data_path, sheet_name=sheet_name)
        print(f"[OK] 成功读取数据：{data_path}，数据形状：{df.shape}")
    except Exception as e:
        raise ValueError(f"[ERROR] 读取Excel失败：{str(e)}")

    # # 过滤某个队列
    # df = df[df['是否在验证集'] == "Yes"]

    # 2. 数据合法性检查
    # 检查分组列存在性
    if group_col not in df.columns:
        raise ValueError(f"[ERROR] 分组列「{group_col}」不存在，数据列：{list(df.columns)}")
    # 检查病例/对照标签存在性
    group_values = df[group_col].unique()
    if case_label not in group_values:
        raise ValueError(f"[ERROR] 病例组标签「{case_label}」不存在，分组值：{list(group_values)}")
    if control_label not in group_values:
        raise ValueError(f"[ERROR] 对照组标签「{control_label}」不存在，分组值：{list(group_values)}")
    # 确定目标变量（排除分组列）
    if target_vars is None:
        target_vars = [col for col in df.columns if col != group_col]
        print(f"未指定目标变量，将统计所有非分组列：{target_vars}")
    else:
        for var in target_vars:
            if var not in df.columns:
                raise ValueError(f"[ERROR] 目标变量{var}不存在，数据列：{list(df.columns)}")

    # 3. 确定变量类型（手动指定优先，否则自动判断）
    var_types = {}
    var_type_specify = var_type_specify or {}
    for var in target_vars:
        if var in var_type_specify:
            var_type = var_type_specify[var].lower()
            if var_type not in ["continuous", "discrete"]:
                raise ValueError(f"[ERROR] 变量{var}类型错误，需为'continuous'或'discrete'")
            var_types[var] = var_type
        else:
            # 自动判断：数值型+唯一值多→连续型；否则离散型
            if pd.api.types.is_numeric_dtype(df[var]):
                var_types[var] = "continuous" if df[var].nunique() > continuous_nunique_thresh else "discrete"
            else:
                var_types[var] = "discrete"
    print(f"变量类型判断结果：{var_types}")

    # 4. 拆分病例组/对照组数据，计算总样本量
    case_df = df[df[group_col] == case_label].copy()
    control_df = df[df[group_col] == control_label].copy()
    n_case, n_control = case_df.shape[0], control_df.shape[0]
    print(f"样本量：病例组(N={n_case})，对照组(N={n_control})")

    # 5. 初始化结果列表
    results = []

    # 6. 逐个处理目标变量
    for var in target_vars:
        var_type = var_types[var]
        # 排除缺失值
        case_data = case_df[var].dropna()
        control_data = control_df[var].dropna()
        print(f"\n处理变量：{var}（类型：{var_type}）")
        print(f"非缺失样本：病例组N={case_data.shape[0]}，对照组N={control_data.shape[0]}")

        # ---------------------- 连续型变量：均值±标准差 + t检验 ----------------------
        if var_type == "continuous":
            # 计算均值和标准差
            case_mean, case_std = case_data.mean(), case_data.std()
            control_mean, control_std = control_data.mean(), control_data.std()
            case_stats = f"{case_mean:.{decimal_places}f}±{case_std:.{decimal_places}f}"
            control_stats = f"{control_mean:.{decimal_places}f}±{control_std:.{decimal_places}f}"

            # 统计检验：先做方差齐性检验（Levene），再做t检验
            try:
                levene_stat, levene_p = levene(case_data, control_data)
                equal_var = levene_p >= 0.05  # 方差齐→equal_var=True
                t_stat, t_p = ttest_ind(case_data, control_data, equal_var=equal_var)
                print(f"[OK] t检验：统计量={t_stat:.4f}，p值={t_p:.4f}（方差齐性：{equal_var}）")
            except Exception as e:
                print(f"[ERROR] 检验失败：{str(e)}")
                t_p = np.nan

            # 格式化p值（加标^a）
            if np.isnan(t_p):
                p_str = "NA"
            else:
                p_str = "<0.001" if t_p < 0.001 else f"{t_p:.{p_decimal_places}f}"
                p_str += "^a"  # t检验标a

            # 添加结果
            results.append({
                "Characteristics": var,
                f"Cases (N={n_case})": case_stats,
                f"Controls (N={n_control})": control_stats,
                "p-value": p_str
            })

        # ---------------------- 离散型变量：计数(占比%) + 卡方/Fisher检验 ----------------------
        else:
            # 制作列联表（行=分组，列=变量分类）
            try:
                grouped_data = pd.concat([
                    pd.DataFrame({"group": case_label, "var": case_data}),
                    pd.DataFrame({"group": control_label, "var": control_data})
                ])
                cross_tab = pd.crosstab(grouped_data["group"], grouped_data["var"])
                print(f"列联表：\n{cross_tab}")
            except Exception as e:
                print(f"[ERROR] 列联表制作失败：{str(e)}")
                continue

            # 统计检验：卡方检验（期望频数不足则用Fisher）
            # try:
            #     chi2_stat, chi2_p, dof, expected = chi2_contingency(cross_tab)
            #     if (expected < 5).any():
            #         print("[WARNING] 存在期望频数<5，使用Fisher精确检验")
            #         fisher_stat, fisher_p = fisher_exact(cross_tab)
            #         test_p, test_name = fisher_p, "Fisher"
            #     else:
            #         test_p, test_name = chi2_p, "Chi-square"
            #     print(f"[OK] {test_name}检验：p值={test_p:.4f}")
            # except Exception as e:
            #     print(f"[ERROR] 检验失败：{str(e)}")
            #     test_p = np.nan

            # 统计检验：卡方检验（期望频数不足则用Fisher）
            try:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(cross_tab)
                # 仅当列联表为 2x2 且期望频数<5 时使用 Fisher 精确检验
                if cross_tab.shape == (2, 2) and (expected < 5).any():
                    print("[WARNING] 存在期望频数<5，使用Fisher精确检验")
                    fisher_stat, fisher_p = fisher_exact(cross_tab)
                    test_p, test_name = fisher_p, "Fisher"
                else:
                    # 非 2x2 或期望频数充足时，直接使用卡方检验
                    if (expected < 5).any():
                        print("[WARNING] 存在期望频数<5，但列联表非2x2，继续使用卡方检验（结果仅供参考）")
                    test_p, test_name = chi2_p, "Chi-square"
                print(f"[OK] {test_name}检验：p值={test_p:.4f}")
            except Exception as e:
                print(f"[ERROR] 检验失败：{str(e)}")
                test_p = np.nan

            # 格式化p值（加标^b）
            if np.isnan(test_p):
                p_str = "NA"
            else:
                p_str = "<0.001" if test_p < 0.001 else f"{test_p:.{p_decimal_places}f}"
                p_str += "^b"  # 卡方/Fisher检验标b

            # 计算每个分类的计数和占比（仅第一个分类显示p值）
            is_first_cat = True
            for cat in cross_tab.columns:
                # 病例组：计数(占比%)
                case_cnt = cross_tab.loc[case_label, cat] if case_label in cross_tab.index else 0
                case_ratio = (case_cnt / n_case) * 100
                # 对照组：计数(占比%)
                control_cnt = cross_tab.loc[control_label, cat] if control_label in cross_tab.index else 0
                control_ratio = (control_cnt / n_control) * 100

                # 添加结果（仅第一个分类显示p值）
                results.append({
                    "Characteristics": f"{var} - {cat}",
                    f"Cases (N={n_case})": f"{case_cnt}({case_ratio:.0f}%)",
                    f"Controls (N={n_control})": f"{control_cnt}({control_ratio:.0f}%)",
                    "p-value": p_str if is_first_cat else ""
                })
                is_first_cat = False

    # 7. 整理结果为DataFrame
    results_df = pd.DataFrame(results)
    print(f"\n[OK] 统计完成！最终表格：\n{results_df.to_string(index=False)}")

    # 8. 输出到Excel（若指定路径）
    if output_path:
        try:
            results_df.to_excel(output_path, index=False, engine="openpyxl")
            print(f"[OK] 结果已保存至：{output_path}")
        except Exception as e:
            print(f"[ERROR] 保存Excel失败：{str(e)}")

    return results_df


# ---------------------- 示例调用（根据你的数据修改参数） ----------------------
if __name__ == "__main__":
    # 1. 配置参数（需根据你的Excel文件修改！）
    CONFIG = {
        "data_path": "COPD核酸质谱基线数据（最新）9.8.xlsx",  # 你的Excel文件路径
        "sheet_name": "Sheet1",  # Excel工作表名称
        "group_col": "分组",  # 分组列名
        "case_label": "病例",  # 病例组标签
        "control_label": "对照",  # 对照组标签
        # 要统计的临床指标（可自由增减）
        "target_vars": ["年龄",
                        "age_cat1",
                        "age_cat2",
                        "age_cat3",
                        "age_cat4",
                        "性别",
                        "BMI",
                        "吸烟状态(never/former/current)",
                        "舒张后FEV1（A/Pd)",
                        "GOLD分级",
                        "舒张后FEV1/FVC",
                        "FVC预计值",
                        "舒张前FVC（L）",
                        "舒张后FVC（L）",
                        "FEV1预计值",
                        "舒张前FEV1(L)",
                        "舒张前FEV1（A/Pd)",
                        "舒张后FEV1（L）",
                        "舒张前PEF",
                        "舒张后PEF",
                        "舒张前MMEF",
                        "舒张后MMEF",
                        "舒张前MEF50",
                        "舒张后MEF50",
                        "舒张前MEF25",
                        "舒张后MEF25",
                        "白细胞",
                        "中性粒细胞比率",
                        "淋巴细胞比率",
                        "单核细胞比率",
                        "嗜酸性粒细胞比率",
                        "嗜碱细胞比率",
                        "有核红细胞比率",
                        "中性粒细胞数",
                        "淋巴细胞数",
                        "单核细胞数",
                        "嗜酸性粒细胞数",
                        "嗜碱性粒细胞数",
                        "有核红细胞绝对值",
                        "红细胞",
                        "血红蛋白",
                        "红细胞压积（比容）",
                        "PLT"
                        ],
        # 手动指定变量类型（可选，自动判断不准时用）
        "var_type_specify": {
            "舒张前MEF25": "continuous",
            "有核红细胞比率": "continuous",
            "有核红细胞绝对值": "continuous",
        },
        "output_path": "临床指标统计结果260909.xlsx"  # 输出文件路径
    }

    # CONFIG = {
    #     "data_path": "prs_copd_251029.xlsx",  # 你的Excel文件路径
    #     "sheet_name": "alias",  # Excel工作表名称
    #     "group_col": "group",  # 分组列名
    #     "case_label": "case",  # 病例组标签
    #     "control_label": "control",  # 对照组标签
    #     # 要统计的临床指标（可自由增减）
    #     "target_vars": [
    #         'rs7041-rs4588'
    #     ],
    #     # 手动指定变量类型（可选，自动判断不准时用）
    #     "var_type_specify": {
    #     },
    #     "output_path": "rs7041-rs4588统计检验251029.xlsx"  # 输出文件路径
    # }

    # 2. 调用函数生成统计表格
    stats_table = generate_clinical_stats_table(**CONFIG)