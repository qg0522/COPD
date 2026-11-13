import argparse
import textwrap
import pandas as pd
from utils import format_genotypes


def parse_arguments():
    out_prefix = "plink_input"
    sheet_name_allele = "位点信息确认"
    sheet_name_genotype = "基因型结果"
    sheet_name_group = "样本分组"
    group_col = "组别"
    invalid_samples = ""
    invalid_variants = ""
    var_map = ""

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description='make .ped and .map file for plink',
                                     epilog=textwrap.dedent('''\
                                             version:    1.0
                                             date:    2025/09/24
                                             contact:    qiangang@digena.com.cn'''))

    parser.add_argument("-i", dest="input_table", help="the input excel table with allele information", required=True)
    parser.add_argument("-s", dest="sheet_name_allele", help="the sheet name for allele information, "
                                                             "default: {sheet_name_allele}", default=sheet_name_allele)
    parser.add_argument("-d", dest="data_set", help=f"the dataset excel table with sample genotypes and "
                                                    f"sample groups",
                        required=True)
    parser.add_argument("-s1", dest="sheet_name_genotype", help=f"the sheet name for genotype information, "
                                                                f"default: {sheet_name_genotype}",
                        default=sheet_name_genotype)
    parser.add_argument("-s2", dest="sheet_name_group", help=f"the sheet name for sample groups, "
                                                             f"default: {sheet_name_group}", default=sheet_name_group)
    parser.add_argument("-g", dest="group_col", help=f"the column name for sample groups, default: {group_col}",
                        default=group_col)
    parser.add_argument("--invalid_samples", dest="invalid_samples", help=f"the comma-separated sample list, "
                                                                          f"default: {invalid_samples}",
                        default=invalid_samples)
    parser.add_argument("--invalid_variants", dest="invalid_variants", help=f"the comma-separated variant list, "
                                                                            f"default: {invalid_variants}",
                        default=invalid_variants)
    parser.add_argument("--var_map", dest="var_map", help=f"rename some variants according to the variant map, i.e. "
                                                          f"var1:new_var1,var2:new_var2, default: {var_map}",
                        default=var_map)
    parser.add_argument("-o", dest="output", help=f"the output prefix for .ped and .map file, "
                                                  f"default: {out_prefix}", default=out_prefix)

    args = parser.parse_args()

    return args


def get_map_dict(var_map):
    d = {}

    for item in str(var_map).split(","):
        key, value = str(item).split(":")
        d[key] = value

    return d


def get_alleles(data_df, sample, rs_list, allele_df):
    alleles = []

    for rs in rs_list:
        rs_alleles = str(data_df.loc[sample, rs]).split("/")
        alleles.extend(rs_alleles)
        # allele_list = str(allele_df.loc[rs, "allele_string"]).split("/")
        # rs_allele_indices = [(allele_list.index(allele) + 1) for allele in rs_alleles]
        # alleles.extend(rs_allele_indices)

    return alleles


def make_ped_file(input_table, sheet_name_allele, data_set, sheet_name_genotype, sheet_name_group, group_col,
                  invalid_samples, invalid_variants, var_map, out_prefix):

    # 读取位点信息
    allele_df = pd.read_excel(input_table, sheet_name=sheet_name_allele, index_col="rs_id", dtype=str)

    # 读取样本分组
    sample_df = pd.read_excel(data_set, sheet_name=sheet_name_group, index_col=0, dtype=str)

    # 去除部分不合格样本
    if invalid_samples:
        sample_df = sample_df.drop(str(invalid_samples).split(","))

    sample_df = sample_df[sample_df[group_col].isin(['阳性', '阴性'])]
    # 去除肌间静脉
    # sample_df = sample_df[~((sample_df['7、发生栓塞静脉'] == "其他（肌间静脉）") & (sample_df[group_col] == "阳性"))]
    sample_df['group'] = sample_df.apply(lambda r: 1 if r[group_col] == '阴性' else 2, axis=1)

    data_df = pd.read_excel(data_set, sheet_name=sheet_name_genotype, index_col=0, dtype=str)

    # 保留合格样本
    data_df = data_df[data_df.index.isin(sample_df.index)]

    # 一些位点重命名
    if var_map:
        var_map_dict = get_map_dict(var_map)
        data_df.rename(var_map_dict, axis=1, inplace=True)
    # data_df.loc[data_df['rs8176719'] == 'A', 'rs8176719'] = '-'
    # data_df.loc[data_df['rs8176719'] == 'CA', 'rs8176719'] = 'C-'

    # 去除一些不合格位点
    rs_list = list(data_df.columns)
    if invalid_variants:
        rs_list = list(data_df.columns.difference(str(invalid_variants).split(",")))

    # 格式化基因型
    for rs_id in rs_list:
        genotypes = sorted(list(set(list(data_df[rs_id]))))
        genotype_format_dict = format_genotypes(genotypes)
        data_df[rs_id] = data_df[rs_id].map(genotype_format_dict)

    # 输出结果
    ped_out = f"{out_prefix}.ped"
    data_out = []
    # header_columns = ['Family ID', 'Within-family ID', 'Within-family ID of father', 'Within-family ID of mother',
    #                   'Sex code', 'Phenotype value']
    # for rs in rs_list:
    #     header_columns.extend([rs, ""])
    for sample in data_df.index:
        group = sample_df.loc[sample, 'group']
        sample_info = [sample, sample, 0, 0, 0, group]
        alleles = get_alleles(data_df, sample, rs_list, allele_df)
        data_out.append(sample_info + alleles)

    # df_out = pd.DataFrame(data_out, columns=header_columns)
    df_out = pd.DataFrame(data_out)
    df_out.to_csv(ped_out, sep="\t", index=False, header=False)

    map_out = f"{out_prefix}.map"
    map_data = []
    for rs_id in rs_list:
        genome_version, chr, pos_range = str(allele_df.loc[rs_id, "location"]).split(":")
        start, end = str(pos_range).split("-")
        map_data.append([chr, rs_id, 0, int(start)])

    df_out = pd.DataFrame(map_data)
    df_out.to_csv(map_out, sep="\t", index=False, header=False)


def main():
    # parse arguments
    args = parse_arguments()

    # make ped file
    make_ped_file(args.input_table, args.sheet_name_allele, args.data_set, args.sheet_name_genotype,
                  args.sheet_name_group, args.group_col,
                  args.invalid_samples, args.invalid_variants, args.var_map, args.output)


if __name__ == '__main__':
    main()
