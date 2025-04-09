from collections import defaultdict, Counter, OrderedDict
import re
import sys

sys.setrecursionlimit(1000000)
from datetime import datetime
import multiprocessing as mp
import string

def print_tree(move_tree, indent=' '):
    for key, value in move_tree.items():
        if isinstance(value, dict):
            print(f'{indent}|- {key}')
            print_tree(value, indent + '|  ')
        elif isinstance(value, tuple):
            print(f'{indent}|- {key}: tuple')
        else:
            print(f'{indent}|- {key}: {value}')


def lcs_similarity(X, Y):
    m, n = len(X), len(Y)
    c = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if X[i - 1] == Y[j - 1]:
                c[i][j] = c[i - 1][j - 1] + 1
            else:
                c[i][j] = max(c[i][j - 1], c[i - 1][j])
    return 2 * c[m][n] / (m + n)

###########################################################
# 改进版 ParsingCache 类
###########################################################
class ParsingCache:
    def __init__(self):
        self.template_tree = {}
        self.template_list = []   # 保存所有模板字符串，索引即模板ID（0开始）
        self.samples = []         # 保存所有样本，格式：{'log': 日志文本, 'template_id': 模板ID, 'idx': 行号, 'log_num': 日志计数}

    def add_templates(self, event_template, normal=True, extra_info=None):
        """
        添加模板：
         - 如果模板已存在则直接返回对应模板ID，否则插入新模板并返回新模板ID。
         - 支持对比相似模板，若相似度高则进行模板合并。
        """
        # 预处理并分词
        template_tokens = message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1
        # 如果无需合并，则直接插入新模板
        if normal or len(extra_info) == 0:
            new_id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return new_id
        # 尝试合并：遍历 extra_info 中提供的相关模板
        max_similarity = 0
        similar_template = None
        for rt in extra_info:
            tokens_rt = rt.split()
            tokens_et = event_template.split()
            if len(tokens_rt) != len(tokens_et):
                continue
            similarity = lcs_similarity(tokens_rt, tokens_et)
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template = rt
        if max_similarity > 0.8 and similar_template:
            success, merged_id = self.modify(similar_template, event_template)
            if success:
                return merged_id
        new_id = self.insert(event_template, template_tokens, len(self.template_list))
        self.template_list.append(event_template)
        return new_id

    def insert(self, event_template, template_tokens, template_id):
        """
        将模板插入模板树中，并返回模板ID。
        """
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            self.template_tree[start_token] = {}
        move_tree = self.template_tree[start_token]
        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                move_tree[token] = {}
            move_tree = move_tree[token]
            tidx += 1
        # 保存模板相关统计信息：有效词数、通配符数、原模板字符串、模板ID
        move_tree["".join(template_tokens)] = (
            sum(1 for s in template_tokens if s != "<*>"),
            template_tokens.count("<*>"),
            event_template,
            template_id
        )
        return template_id

    def modify(self, similar_template, event_template):
        """
        对比已有模板和新模板，利用最长公共子序列合并生成新模板，
        成功则删除原模板，插入合并后的模板并更新模板列表。
        """
        similar_tokens = similar_template.split()
        event_tokens = event_template.split()
        merged_tokens = []
        for t1, t2 in zip(similar_tokens, event_tokens):
            if t1 == t2:
                merged_tokens.append(t1)
            else:
                merged_tokens.append("<*>")
        merged_template = " ".join(merged_tokens)
        success, old_id = self.delete(similar_template)
        if not success:
            return False, -1
        self.insert(merged_template, message_split(merged_template), old_id)
        self.template_list[old_id] = merged_template
        return True, old_id

    def delete(self, event_template):
        """
        从模板树中删除指定模板，并返回删除的模板ID。
        """
        template_tokens = message_split(event_template)
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            return False, -1
        move_tree = self.template_tree[start_token]
        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                return False, -1
            move_tree = move_tree[token]
            tidx += 1
        key = "".join(template_tokens)
        if key not in move_tree:
            return False, -1
        old_id = move_tree[key][3]
        del move_tree[key]
        return True, old_id

    def match_event(self, log):
        """
        利用模板树对日志进行匹配，返回匹配结果。
        """
        return tree_match(self.template_tree, log)

    def add_sample_to_template(self, log: str, template_id: int, idx: int) -> None:
        """
        添加一条日志样本记录到 samples 列表中。
        """
        if not hasattr(self, 'samples'):
            self.samples = []
        sample = {'log': log, 'template_id': template_id, 'idx': idx, 'log_num': 1}
        self.samples.append(sample)

    def get_samples_by_template(self, template_id: int) -> list:
        """
        返回所有模板ID为 template_id 的样本记录（相当于 select_samples_by_template_id）。
        """
        return [s for s in self.samples if s['template_id'] == template_id]

    def select_sample_len_by_template(self, template_id: int) -> int:
        """
        返回模板 template_id 下所有样本的日志计数总和（相当于 select_sample_len_by_template_id）。
        """
        return sum(s.get('log_num', 1) for s in self.get_samples_by_template(template_id))

    def update_sample_template(self, sample: dict, new_template_id: int) -> None:
        """
        更新样本记录中所属模板ID。
        """
        sample['template_id'] = new_template_id

    def update_sample_log(self, sample: dict) -> None:
        """
        更新样本日志计数（增加1）。
        """
        sample['log_num'] += 1

###########################################################
# 辅助函数
###########################################################
def message_split(message):
    """
    将日志或模板文本进行分词处理，同时对特殊字符进行处理，返回 token 列表。
    """
    punc = "!\"#$%&'()+,-/:;=?@.[\\]^_`{|}~"
    splitter_regex = re.compile("([{}])".format(re.escape(punc)))
    tokens = re.split(splitter_regex, message)
    tokens = [token.strip() for token in tokens if token.strip() != ""]
    # 简单合并连续的通配符
    filtered_tokens = []
    for token in tokens:
        if token == "<*>" and filtered_tokens and filtered_tokens[-1] == "<*>":
            continue
        filtered_tokens.append(token)
    return filtered_tokens

def lcs_similarity(X, Y):
    """
    计算两个 token 序列之间的最长公共子序列相似度（归一化为 [0,2] 区间，再映射到 [0,1]）。
    """
    m, n = len(X), len(Y)
    c = [[0]*(n+1) for _ in range(m+1)]
    for i in range(1, m+1):
        for j in range(1, n+1):
            if X[i-1] == Y[j-1]:
                c[i][j] = c[i-1][j-1] + 1
            else:
                c[i][j] = max(c[i][j-1], c[i-1][j])
    return 2 * c[m][n] / (m + n) if m+n > 0 else 0

def tree_match(match_tree, log_content):
    """
    利用模板树对日志进行匹配，返回匹配结果：(模板字符串, 模板ID, 额外信息) 或 ("NoMatch", "NoMatch", relevant_templates)。
    """
    log_tokens = message_split(log_content)
    template, template_id, parameter_str = match_template(match_tree, log_tokens)
    if template:
        return (template, template_id, parameter_str)
    else:
        return ("NoMatch", "NoMatch", parameter_str)

def match_template(match_tree, log_tokens):
    results = []
    find_results = find_template(match_tree, log_tokens, results, [], 1)
    relevant_templates = find_results[1]
    new_results = results if len(results) > 0 else []
    if len(new_results) > 0:
        if len(new_results) > 1:
            new_results.sort(key=lambda x: (-x[1][0], x[1][1]))
        return new_results[0][1][2], new_results[0][1][3], new_results[0][2]
    return False, False, relevant_templates

def get_all_templates(move_tree):
    result = []
    for key, value in move_tree.items():
        if isinstance(value, tuple):
            result.append(value[2])
        else:
            result.extend(get_all_templates(value))
    return result

def find_template(move_tree, log_tokens, result, parameter_list, depth):
    flag = 0  # 0：无匹配，1：待继续，2：匹配成功
    if len(log_tokens) == 0:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.append((key, value, tuple(parameter_list)))
                flag = 2
        if "<*>" in move_tree:
            parameter_list.append("")
            subtree = move_tree["<*>"]
            if isinstance(subtree, tuple):
                result.append(("<*>", subtree, tuple(parameter_list)))
                flag = 2
            else:
                for key, value in subtree.items():
                    if isinstance(value, tuple):
                        result.append((key, value, tuple(parameter_list)))
                        flag = 2
        # 返回匹配结果
    else:
        token = log_tokens[0]
        relevant_templates = []
        if token in move_tree:
            subflag, subrel = find_template(move_tree[token], log_tokens[1:], result, parameter_list, depth+1)
            if subflag:
                flag = 2
            else:
                flag = max(flag, 1)
                relevant_templates.extend(subrel)
        if "<*>" in move_tree:
            subtree = move_tree["<*>"]
            idx = 0
            while idx < len(log_tokens):
                parameter_list.append("".join(log_tokens[:idx]))
                subflag, subrel = find_template(subtree, log_tokens[idx:], result, parameter_list, depth+1)
                if subflag:
                    flag = 2
                else:
                    flag = max(flag, 1)
                    relevant_templates.extend(subrel)
                if parameter_list:
                    parameter_list.pop()
                idx += 1
        if flag == 2:
            return (True, [])
        elif flag == 1:
            return (False, relevant_templates)
        else:
            if depth >= 2:
                return (False, get_all_templates(move_tree))
            else:
                return (False, [])