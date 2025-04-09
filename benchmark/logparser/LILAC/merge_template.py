from typing import List
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Template import Template
from logparser.utils.utils_template import tokenize_text_with_punctuation_regex, com_part_to_tokenized, prune_template_by_distance, \
    get_matrix_of_distance, get_cluster, all_common_substrings
from logparser.utils.utils_match import unique_wildcard, check_match_all




def get_similar_templates(templates: List[Template], new_template: Template, cluster_enable: bool) -> List[Template]:
    """用编辑记录进行聚类， 获取该模板的相似模板"""
    templates = prune_template_by_distance(templates, new_template, 0.3)
    templates.append(new_template)
    similar_templates = templates

    if cluster_enable:
        if len(similar_templates) == 1:
            return similar_templates
        dist_matrix = get_matrix_of_distance(templates)
        cluster_labels = get_cluster(dist_matrix)

        res_label = cluster_labels[-1]  # 该模型所在的聚类标签
        similar_templates = [templates[i] for i in range(len(templates)) if cluster_labels[i] == res_label]  # 该模板的所有相似模板（包括该模板）
    return similar_templates


def find_subarray_indices(lst, subarray, start_index):
    """查找块"""
    sub_len = len(subarray)
    for i in range(start_index, len(lst) - sub_len + 1):
        if lst[i:i + sub_len] == subarray:
            return i


def find_sequence_indices(lst, sequence):
    """查找公共部分和私有部分"""
    indices = []
    start_index = 0
    for item in sequence:
        index = find_subarray_indices(lst, item, start_index)  # 查找块
        indices.append(''.join(lst[start_index:index]))
        indices.append(''.join(item))
        start_index = index + len(item)
    indices.append(''.join(lst[start_index:]))
    return indices


def merge_templates_by_lcs(templates: List[Template]) -> Template:
    """合并多个模板，基于 LCS 方法实现"""
    if len(templates) == 1:
        return templates[0]

    # 对所有模板的文本进行分词
    templates_tokenized = [tokenize_text_with_punctuation_regex(template.template_text) for template in templates]

    # 计算第一个和第二个模板的公共子序列块
    com_part = all_common_substrings(templates_tokenized[0], templates_tokenized[1])
    # 针对后续模板，依次更新公共子序列块
    for template_tokenized in templates_tokenized[2:]:
        com_part = com_part_to_tokenized(com_part)
        com_part = all_common_substrings(com_part, template_tokenized)

    # 将公共部分合并为字符串，并用 <*> 连接
    coms = [''.join(com) for com in com_part]
    fixed = '<*>'.join(''.join(com) for com in coms)

    # 检查合并后的模板是否能匹配所有原始模板
    templates_text = [template.template_text for template in templates]
    if not check_match_all(fixed, templates_text):
        if check_match_all('<*>' + fixed, templates_text):
            fixed = '<*>' + fixed
        elif check_match_all(fixed + '<*>', templates_text):
            fixed = fixed + '<*>'
        else:
            fixed = '<*>' + fixed + '<*>'

    fixed = unique_wildcard(fixed)

    # 过滤掉 None 的 sample_limit，并提供默认值（这里默认使用 0）
    min_sample_limit_candidates = [template.sample_limit for template in templates if template.sample_limit is not None]
    if min_sample_limit_candidates:
        min_sample_limit = min(min_sample_limit_candidates)
    else:
        min_sample_limit = 0  # 或者设置为其他合理的默认值，比如 50

    merged_template = Template(fixed, sample_limit=min_sample_limit)
    return merged_template

