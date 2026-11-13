from sklearn.metrics import roc_curve, auc
import numpy as np
import itertools


def format_genotypes(genotypes):
    genotype_dict = {}

    is_dot_found = False
    is_het_found = False
    alleles = []
    for genotype in genotypes:
        if '.' in genotype:
            is_dot_found = True
            is_het_found = True
            alleles = str(genotype).split(".")
            break

    if not is_dot_found:
        for genotype in genotypes:
            if len(genotype) == 2:
                is_het_found = True
                alleles = list(genotype)
                break

    if is_het_found:
        for genotype in genotypes:
            if genotype in alleles:
                genotype_dict[genotype] = f"{genotype}/{genotype}"
            else:
                genotype_dict[genotype] = "/".join(alleles)
    else:
        for genotype in genotypes:
            genotype_dict[genotype] = f"{genotype}/{genotype}"

    return genotype_dict


def roc_plot(y_true, y_score, method, ax):

    neg_sample_num = list(y_true).count(0)
    pos_sample_num = list(y_true).count(1)
    if neg_sample_num > 0:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        optimal_idx = int(np.argmax(tpr - fpr))
        optimal_threshold = round(thresholds[optimal_idx], 4)
        # threshold_range = get_threshold_range(list(y_score), optimal_idx)
        optimal_fpr = fpr[optimal_idx]
        optimal_tpr = tpr[optimal_idx]
        figure_name = ""
        ax.plot(fpr, tpr, lw=1, label='{0:s} AUC = {1:.2f}, opt cutoff = {2:.4f}'.
                 format(figure_name, roc_auc, optimal_threshold))  # 画出当前分割数据的ROC曲线
        ax.scatter(optimal_fpr, optimal_tpr, marker='o')
        ax.text(optimal_fpr, optimal_tpr, '(%.2f, %.2f)' % (optimal_fpr, optimal_tpr), color='red')

        ax.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))  # 画对角线

        ax.set_xlim([-0.05, 1.05])  # 设置x、y轴的上下限，设置宽一点，以免和边缘重合，可以更好的观察图像的整体
        ax.set_ylim([-0.05, 1.05])
        ax.set_xlabel('1-specificity / false positive rate')
        ax.set_ylabel('sensitivity / recall')  # 可以使用中文，但需要导入一些库即字体
        ax.set_title(method)
        ax.legend(loc="lower right", fontsize="small")

        stat_data = [method, pos_sample_num, neg_sample_num, round(optimal_tpr, 2), round(1 - optimal_fpr, 2),
                     round(roc_auc, 2), optimal_threshold]
    else:
        optimal_tpr = round((len(y_score) - list(y_score).count(0)) / len(y_score), 2)
        specifity = "na"
        roc_auc = "na"
        threshold_range = f"[{min(y_score)}, {max(y_score)}]"
        stat_data = [method, pos_sample_num, neg_sample_num, optimal_tpr, specifity, roc_auc, threshold_range]

    return stat_data


def get_genotype(alleles, allele1, allele2):

    if allele1 == allele2:
        return f"{allele1}/{allele2}"

    if alleles.index(allele1) < alleles.index(allele2):
        return f"{allele1}/{allele2}"
    else:
        return f"{allele2}/{allele1}"


def get_dict_value(d, k):
    v = 0

    if k in d:
        v = d[k]

    return v

def get_all_combinations_two_rounds_simple(lists):
    """
    简化版：列出从多个等长列表中每次每个取一个元素，取两遍（不能放回）的所有可能性。

    参数:
        lists: 列表的列表，每个列表等长。

    返回:
        一个列表，每个元素是一个元组(first, second)，表示两次抽取的元素组合。
    """
    if not lists:
        return []

    n = len(lists[0])
    k = len(lists)

    # 验证所有列表等长
    if any(len(lst) != n for lst in lists):
        raise ValueError("所有列表必须等长")

    # 生成所有可能的两次抽取组合
    results = []
    for first_indices in itertools.product(range(n), repeat=k):
        # 计算每个列表的剩余索引
        remaining_indices = [
            [j for j in range(n) if j != first_indices[i]]
            for i in range(k)
        ]

        # 生成第二次抽取的所有可能
        for second_indices in itertools.product(*remaining_indices):
            first_selection = tuple(lists[i][first_indices[i]] for i in range(k))
            second_selection = tuple(lists[i][second_indices[i]] for i in range(k))
            results.append((first_selection, second_selection))

    return results