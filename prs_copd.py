import argparse
import math
import random
import sys
import textwrap
from collections import Counter
from enum import Enum
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np
from utils import format_genotypes, roc_plot, get_all_combinations_two_rounds_simple
random.seed(100)


def parse_arguments():
    out_file = "./prs_dataset.xlsx"
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='polygenic risk score for the copd dataset',
                                     epilog=textwrap.dedent('''\
                                             version:    1.0
                                             date:    2025/09/04
                                             contact:    qiangang@digena.com.cn'''))

    parser.add_argument("-i", dest="input_table", help="the input excel table with allele information", required=True)
    parser.add_argument("-s", dest="sheet_name", help="the sheet name for allele information", required=True)
    parser.add_argument("-d", dest="data_set", help=f"the dataset excel table with sample genotypes and "
                                                    f"sample groups",
                        required=True)
    parser.add_argument("-o", dest="output", help="the output result, default: %s" % out_file, default=out_file)

    args = parser.parse_args()

    return args


class PrsMethod(Enum):
    # CGR = 1
    # CGR_sum = 2
    # CGR_gy = 3
    # EV_GRS = 4
    # PRSice = 5
    PRS = 1


def get_sample_prs(rs_info, data_df, rs_list, sample):
    prs_dict = {
        # PrsMethod.CGR.name: 1,
        # PrsMethod.CGR_sum.name: 0,
        # PrsMethod.CGR_gy.name: 0,
        # PrsMethod.EV_GRS.name: 0,
        PrsMethod.PRS.name: 0
    }

    rs_genotype = {}
    rs_genotype_alias = {}

    # rs_num = len(rs_list)
    # score = 0
    for rs_id_comb in rs_info:
        rs_obj = rs_info[rs_id_comb]
        if rs_obj.effect_allele is not None:
            sample_genotype = data_df.loc[sample, rs_id_comb]
            allele_count_dict = Counter(str(sample_genotype).split("/"))
            effect_allele_count = 0
            if rs_obj.effect_allele in allele_count_dict:
                effect_allele_count = allele_count_dict[rs_obj.effect_allele]
            # non_effect_allele_count = 2 - effect_allele_count

            rs_genotype[rs_id_comb] = [sample_genotype, effect_allele_count]

        # if effect_allele_count > 0:
        #     sOR = effect_allele_count * rs_obj.odds_ratio
        # else:
        #     sOR = 1
        # pOR = rs_obj.pop_odds_ratio

        # maf = 0.0
        # if rs_obj.effect_allele in rs_obj.pop_freq:
        #     maf = rs_obj.pop_freq[rs_obj.effect_allele]

        # log_odds_ratio = math.log(rs_obj.odds_ratio)
        # wEV = log_odds_ratio * math.sqrt(2 * maf * (1 - maf))
        # prs_dict[PrsMethod.CGR.name] *= sOR / pOR
        # prs_dict[PrsMethod.CGR_sum.name] += sOR / pOR
        # prs_dict[PrsMethod.EV_GRS.name] += wEV * effect_allele_count
            if effect_allele_count > 0:
                if rs_obj.odds_ratio > 1:
                    prs_dict[PrsMethod.PRS.name] += rs_obj.odds_ratio * effect_allele_count
                else:
                    prs_dict[PrsMethod.PRS.name] += max(0, 1 - (1 - rs_obj.odds_ratio) * effect_allele_count)
            else:
                prs_dict[PrsMethod.PRS.name] += 1

        else:
            if rs_obj.is_haplotype:
                rs_ids = str(rs_id_comb).split('-')
                sample_genotype_list = []
                for rs_id in rs_ids:
                    sample_genotype = data_df.loc[sample, rs_id]
                    sample_genotype_list.append(str(sample_genotype).split("/"))

                haplotype_info_list = get_all_combinations_two_rounds_simple(sample_genotype_list)

                is_genotype_found = False
                for haplotype_info in haplotype_info_list:
                    hap1 = "-".join(haplotype_info[0])
                    hap2 = "-".join(haplotype_info[1])
                    genotype = "/".join([hap1, hap2])
                    if genotype in rs_obj.genotype_odds_ratio_dict:
                        odds_ratio = rs_obj.genotype_odds_ratio_dict[genotype]
                        prs_dict[PrsMethod.PRS.name] += odds_ratio
                        is_genotype_found = True
                        rs_genotype_alias[rs_id_comb] = rs_obj.genotype_alias[genotype]
                        break

                if not is_genotype_found:
                    sys.stderr.write(f"[ERROR]: can't find odds ratio for {rs_id_comb} of sample {sample}\n")
                    sys.exit(1)

            else:
                rs_ids = [rs_id_comb]
                sample_genotype = data_df.loc[sample, rs_id_comb]
                if sample_genotype in rs_obj.genotype_odds_ratio_dict:
                    odds_ratio = rs_obj.genotype_odds_ratio_dict[sample_genotype]
                    prs_dict[PrsMethod.PRS.name] += odds_ratio

            for rs_id in rs_ids:
                sample_genotype = data_df.loc[sample, rs_id]
                rs_genotype[rs_id] = [sample_genotype, 0]

        # if sample in ['VTE-3', 'VTE-139']:
        #     gt_score = rs_obj.odds_ratio * effect_allele_count
        #     score += gt_score
        #     print(f"{sample}\t{rs_id}\t{rs_obj.effect_allele}\t{rs_obj.odds_ratio}\t{effect_allele_count}\t"
        #           f"{gt_score}\t{score}")

    # prs_dict[PrsMethod.CGR_gy.name] = prs_dict[PrsMethod.CGR_sum.name] - rs_num + 1

    return prs_dict, rs_genotype, rs_genotype_alias


class Rs:
    def __init__(self, rs_id):
        self.rs_id = rs_id
        self.ref_allele = None
        self.effect_allele = None
        self.non_effect_allele = None
        self.odds_ratio = None
        self.pop_odds_ratio = 0
        self.pop_freq = {}
        self.genotype_odds_ratio_dict = {}
        self.genotype_alias = {}
        self.is_haplotype = False

    def rs_id_list(self):
        if '-' in self.rs_id:
            return str(self.rs_id).split("-")
        else:
            return [self.rs_id]


def get_rs_info(df):
    rs_info = {}

    assert isinstance(df, pd.DataFrame)
    for index, row in df.iterrows():
        rs_id = row['rs_id']

        if rs_id not in rs_info:
            rs_obj = Rs(rs_id)
        else:
            rs_obj = rs_info[rs_id]

        if '-' in rs_id:
            rs_obj.is_haplotype = True

        allele_type = row['type']
        if allele_type == "allele":
            rs_obj.effect_allele = row['effect']
            rs_obj.non_effect_allele = row['non_effect']
            if pd.isna(row['OR']):
                rs_obj.odds_ratio = 1
            else:
                rs_obj.odds_ratio = float(row['OR'])
        elif allele_type == "genotype":
            effect_genotypes = str(row['effect']).split(' + ')
            non_effect_genotypes = str(row['non_effect']).split(' + ')
            effect_odds_ratio = float(row['OR'])
            non_effect_odds_ratio = 1.0
            geno_alias_effect, geno_alias_noneffect = row['模型'].split(' vs. ')
            for i in [1, 2]:
                if i == 1:
                    genotypes = effect_genotypes
                    odds_ratio = effect_odds_ratio
                else:
                    genotypes = non_effect_genotypes
                    odds_ratio = non_effect_odds_ratio

                for genotype in genotypes:
                    allele1, allele2 = str(genotype).split("/")
                    rev_genotype = f"{allele2}/{allele1}"
                    for geno in [genotype, rev_genotype]:
                        if geno not in rs_obj.genotype_odds_ratio_dict:
                            rs_obj.genotype_odds_ratio_dict[geno] = odds_ratio
                        if geno not in rs_obj.genotype_alias:
                            rs_obj.genotype_alias[geno] = geno_alias_effect if i == 1 else geno_alias_noneffect
        else:
            sys.stderr.write(f"[ERROR]: unknown allele type: {allele_type}\n")
            sys.exit(1)

        rs_info[rs_id] = rs_obj

    return rs_info


def prs_dataset(input_table, sheet_name, data_set, output):

    # read allele information
    df = pd.read_excel(input_table, sheet_name=sheet_name)
    assert isinstance(df, pd.DataFrame)
    # df = df.dropna(subset=['OR值'])

    # # 过滤掉某些位点
    # df = df[~df['rs_id'].isin(["rs7041-rs4588"])]

    # 筛选某些5个差异显著位点
    # df = df[df['rs_id'].isin(['rs1800925', 'rs1800629', 'rs8034191', 'rs16969968', 'rs1051730'])]

    # read sample groups and genotypes
    sample_sheet_name = "分组信息"
    sample_df = pd.read_excel(data_set, sheet_name=sample_sheet_name, index_col=0, dtype=str)

    sample_df = sample_df[sample_df['分组'].isin(['病例', '对照'])]
    sample_df['group'] = sample_df.apply(lambda r: 'control' if r['分组'] == '对照' else 'case', axis=1)

    data_sheet_name = "质谱检测结果"
    data_df = pd.read_excel(data_set, sheet_name=data_sheet_name, index_col=0, dtype=str)

    # 保留合格样本
    data_df = data_df[data_df.index.isin(sample_df.index)]

    rs_list = list(data_df.columns)

    # format genotypes
    for rs_id in rs_list:
        genotypes = sorted(list(set(list(data_df[rs_id]))))
        genotype_format_dict = format_genotypes(genotypes)
        data_df[rs_id] = data_df[rs_id].map(genotype_format_dict)

    # get sample PRS
    prs_methods = [m.name for m in PrsMethod]
    header0 = ['group', 'sample_index']
    header = header0 + prs_methods
    data = []
    feat_data = []
    geno_data = []
    alias_data = []
    groups = ['case', 'control']

    rs_info = get_rs_info(df)
    rs_list_from_info = []
    rs_list_hap = []
    for rs in rs_info:
        rs_obj = rs_info[rs]
        assert isinstance(rs_obj, Rs)
        if rs_obj.is_haplotype:
            rs_list_hap.append(rs)
        rs_list_from_info.extend(rs_info[rs].rs_id_list())
    rs_list = sorted(list(set(rs_list_from_info).intersection(set(rs_list))))

    feat_header = header0 + rs_list
    geno_header = header0 + rs_list
    alias_header = header0 + rs_list_hap
    for group in groups:
        samples = list(sample_df[sample_df['group'] == group].index)
        for i in range(len(samples)):
            print(f"group: {group}, sample: {i + 1}")
            sample = str(samples[i])
            prs_dict, rs_genotype_dict, rs_genotype_alias_dict = get_sample_prs(rs_info, data_df, rs_list, sample)
            prs_values = [prs_dict[m] for m in prs_methods]
            genotypes = [rs_genotype_dict[i][0] for i in rs_list]
            genotype_alias_list = [rs_genotype_alias_dict[i] for i in rs_list_hap]
            effect_allele_count_list = [rs_genotype_dict[i][1] for i in rs_list]
            data.append([group, sample] + prs_values)
            geno_data.append([group, sample] + genotypes)
            feat_data.append([group, sample] + effect_allele_count_list)
            alias_data.append([group, sample] + genotype_alias_list)

    # add more PRS methods
    out_df = pd.DataFrame(data, columns=header)
    # CGR_min = min(list(out_df[PrsMethod.CGR.name]))
    # CGR_max = max(list(out_df[PrsMethod.CGR.name]))
    # GRS_min = min(list(out_df[PrsMethod.EV_GRS.name]))
    # GRS_max = max(list(out_df[PrsMethod.EV_GRS.name]))
    # out_df['CGR_norm'] = out_df.apply(lambda r: (r[PrsMethod.CGR.name] - CGR_min) / (CGR_max - CGR_min) * 10,
    #                                   axis=1)
    # out_df['EV_GRS_norm'] = out_df.apply(lambda r: (r[PrsMethod.EV_GRS.name] - GRS_min) / (GRS_max - GRS_min) * 10,
    #                                      axis=1)
    # out_df['SCDRisk'] = out_df.apply(lambda r: 0.5 * (r['CGR_norm'] + r['EV_GRS_norm']), axis=1)

    # filter outliner samples
    # outliner_quantile = 0.15
    # outliner_percent = 0.8
    # select_method = 'PRS'
    # lower_quantile = out_df[select_method].quantile(outliner_quantile)
    # upper_quantile = out_df[select_method].quantile(1 - outliner_quantile)
    # print("filter outliners ...")
    # print(f"lower_quantile: {lower_quantile}, upper_quantile: {upper_quantile}")
    # case_outliners = list(out_df[(out_df[select_method] < lower_quantile) & (out_df['group'] == 'case')].index)
    # case_outliners = random.sample(case_outliners, int(len(case_outliners) * outliner_percent))
    # control_outliners = list(out_df[(out_df[select_method] > upper_quantile) & (out_df['group'] == 'control')].index)
    # control_outliners = random.sample(control_outliners, int(len(control_outliners) * outliner_percent))
    # left_index = list(set(out_df.index) - set(case_outliners) - set(control_outliners))
    # out_df = out_df[out_df.index.isin(left_index)]

    # get histogram
    out_prefix = str(output).replace(".xlsx", "")
    sns.set(style="whitegrid")
    methods = out_df.columns[2:]
    row_num = 2
    col_num = 2
    fig, ax = plt.subplots(row_num, col_num, figsize=(8, 6), layout='tight')
    bin_num = 50
    hist_data = []
    hist_header = ['PRS_method', 'interval_index', 'interval']
    for group in groups:
        hist_header += [f'{group} count', f'{group} %',
                        f'{group} count cumu', f'{group} % cumu', ]
    hist_header.extend(['incidence', 'incidence cumu'])
    for index, m in enumerate(methods):
        rn = math.ceil((index + 1) / col_num) - 1
        if (index + 1) % col_num == 0:
            cn = col_num - 1
        else:
            cn = (index + 1) % col_num - 1

        hist, bin_edges = np.histogram(out_df[m], bins=bin_num)
        group_count_cumulative_dict = {'case': 0, 'control': 0}
        bin_count_cumulative = 0
        for i in range(len(bin_edges) - 1):
            bin_s = bin_edges[i]
            bin_e = bin_edges[i + 1]
            group_counts = []
            group_count_dict = {'case': 0, 'control': 0}
            bin_count = out_df[m].between(bin_s, bin_e).sum()
            bin_count_cumulative += bin_count
            for group in groups:
                group_df = out_df[out_df['group'] == group]
                total_count = group_df[m].count()
                count = group_df[m].between(bin_s, bin_e).sum()
                group_count_dict[group] = count
                group_count_cumulative_dict[group] += count
                percent = round(count / total_count * 100, 4)
                percent_cumulative = round(group_count_cumulative_dict[group] / total_count * 100, 4)
                group_counts.extend([count, percent])
                group_counts.extend([group_count_cumulative_dict[group], percent_cumulative])
            if bin_count > 0:
                incidence_percent = round(group_count_dict['case'] / bin_count * 100, 4)
            else:
                incidence_percent = 0.0
            if bin_count_cumulative > 0:
                incidence_percent_cumulative = round(group_count_cumulative_dict['case'] / bin_count_cumulative * 100, 4)
            else:
                incidence_percent_cumulative = 0.0
            hist_data.append([m, i+1, [round(bin_s, 4), round(bin_e, 4)]] + group_counts +
                             [incidence_percent, incidence_percent_cumulative])

        sns.histplot(data=out_df, x=m, hue='group', stat="count", bins=bin_num, ax=ax[rn, cn], kde=True)
        # ax[rn, cn].legend(loc='upper right', fontsize='small')

    fig.savefig(f"{out_prefix}_hist.pdf")

    # roc plot
    roc_data = []
    roc_headers = ["PRS_method", "expect_pos_num", "expect_neg_num", "sensitivity", "specificity", "AUC", "cutoff"]
    y_true = [1 if x == 'case' else 0 for x in list(out_df['group'])]
    fig, ax = plt.subplots(row_num, col_num, figsize=(8, 6), layout='tight')
    for index, m in enumerate(methods):
        rn = math.ceil((index + 1) / col_num) - 1
        if (index + 1) % col_num == 0:
            cn = col_num - 1
        else:
            cn = (index + 1) % col_num - 1
        y_score = out_df[m]
        stat_data = roc_plot(y_true, y_score, m, ax[rn, cn])
        roc_data.append(stat_data)
    fig.savefig(f"{out_prefix}_roc.pdf")

    # output the result
    hist_df = pd.DataFrame(hist_data, columns=hist_header)
    roc_df = pd.DataFrame(roc_data, columns=roc_headers)
    geno_df = pd.DataFrame(geno_data, columns=geno_header)
    feat_df = pd.DataFrame(feat_data, columns=feat_header)
    alias_df = pd.DataFrame(alias_data, columns=alias_header)
    excel_writer = pd.ExcelWriter(output)
    out_df.to_excel(excel_writer, 'score', index=False)
    hist_df.to_excel(excel_writer, 'hist', index=False)
    roc_df.to_excel(excel_writer, 'roc', index=False)
    geno_df.to_excel(excel_writer, 'geno', index=False)
    feat_df.to_excel(excel_writer, 'feat', index=False)
    alias_df.to_excel(excel_writer, 'alias', index=False)
    excel_writer.close()


def main():
    # parse arguments
    args = parse_arguments()

    # perform PRS calculation for a dataset
    prs_dataset(args.input_table, args.sheet_name, args.data_set, args.output)


if __name__ == '__main__':
    main()
