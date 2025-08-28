import os
from collections import defaultdict, Counter, OrderedDict
import re
import sys
from sentence_transformers import SentenceTransformer
import numpy as np
from collections import defaultdict
from sklearn.cluster import DBSCAN

sys.setrecursionlimit(1000000)
from datetime import datetime
import multiprocessing as mp
import string

# 新增参数配置（在类定义前）
CONFIDENCE_THRESHOLD = 0.8  # 置信度阈值
SEMANTIC_THRESHOLD = 0.3  # 语义相似度阈值
SIMILARITY_THRESHOLD = 0.7  # 结构相似度阈值
CONFIDENCE_WEIGHT = 0.5  # 置信度权重系数
MIN_CLUSTER_SIZE = 1  # 最小聚类数量

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


class ParsingCache(object):
    def __init__(self):
        self.template_tree = {}
        self.template_list = []
        # 新增：置信度存储
        self.confidences = {}
        # 新增：语义模型（延迟加载）
        self.semantic_model = None



    # 新增方法：获取所有模板及其置信度
    def get_all_templates_with_confidence(self):
        """获取缓存中所有模板及其置信度"""
        return [(self.template_list[id], self.confidences.get(id, 1.0))
                for id in range(len(self.template_list))]

    # 新增方法：存储模板置信度
    def store_confidence(self, template_id, confidence):
        """存储模板置信度"""
        self.confidences[template_id] = confidence

    # 新增方法：语义嵌入近邻搜索
    def find_semantic_neighbors(self, new_template, threshold=SEMANTIC_THRESHOLD):
        # return []
        """在嵌入空间中查找相似模板"""
        if self.semantic_model is None:
            try:
                # 从本地路径加载模型
                model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
                # 确保所有必要文件存在
                required_files = [
                    "config.json", "pytorch_model.bin", "sentence_bert_config.json",
                    "tokenizer.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
                ]
                missing_files = []
                for file in required_files:
                    if not os.path.exists(os.path.join(model_path, file)):
                        missing_files.append(file)
                if missing_files:
                    print(f"Missing model files: {', '.join(missing_files)}")
                    print(
                        "Please download all required files from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main")
                    self.semantic_model = False  # Mark as failed
                    return []
                # 加载本地模型
                self.semantic_model = SentenceTransformer(model_path)
                print("Successfully loaded model from local path")
            except Exception as e:
                print(f"Error loading model from local: {e}")
                self.semantic_model = False  # Mark as failed
                return []

        if self.semantic_model is False:
            return []  # Return empty if loading failed

        # 获取所有模板
        all_templates = [tpl for tpl, _ in self.get_all_templates_with_confidence()]

        # 计算嵌入向量
        embeddings = self.semantic_model.encode([new_template] + all_templates)
        new_embedding = embeddings[0]
        template_embeddings = embeddings[1:]

        # 计算欧氏距离
        neighbors = []
        for i, emb in enumerate(template_embeddings):
            distance = np.linalg.norm(new_embedding - emb)
            if distance < threshold:
                neighbors.append((all_templates[i], self.confidences.get(i, 1.0)))
        return neighbors

    # 新增方法：置信度加权相似度计算
    def weighted_similarity_grouping(self, new_template, new_conf, candidates, threshold=SIMILARITY_THRESHOLD):
        # return self.structural_similarity_grouping(new_template, candidates, threshold)
        """计算置信度加权结构相似度"""
        candidate_group = []
        new_tokens = new_template.split()

        for cand_template, cand_conf in candidates:
            # 计算结构相似度
            cand_tokens = cand_template.split()
            lcs_length = self.lcs_length(new_tokens, cand_tokens)
            max_len = max(len(new_tokens), len(cand_tokens))
            struct_sim = lcs_length / max_len if max_len > 0 else 0

            # 置信度加权
            conf_diff = abs(new_conf - cand_conf)
            weighted_sim = struct_sim * (1 - CONFIDENCE_WEIGHT * conf_diff)

            if weighted_sim >= threshold:
                candidate_group.append(cand_template)
        return candidate_group

    # 新增方法：LCS长度计算
    def lcs_length(self, X, Y):
        """计算两个token序列的最长公共子序列长度"""
        m, n = len(X), len(Y)
        c = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if X[i - 1] == Y[j - 1]:
                    c[i][j] = c[i - 1][j - 1] + 1
                else:
                    c[i][j] = max(c[i][j - 1], c[i - 1][j])
        return c[m][n]

    # 新增方法：模板合并
    def merge_templates(self, templates):
        # return templates[0] if templates else ""
        """合并一组语义相似的模板"""
        token_sequences = [t.split() for t in templates]
        max_len = max(len(tokens) for tokens in token_sequences)
        merged = []

        # 按列处理token
        for col_idx in range(max_len):
            col_tokens = []
            for tokens in token_sequences:
                if col_idx < len(tokens):
                    col_tokens.append(tokens[col_idx])

            # 判断是否保留固定token
            if self.all_same(col_tokens):
                merged.append(col_tokens[0])
            else:
                merged.append("<*>")

        return " ".join(merged)

    # 新增方法：判断token是否全部相同
    def all_same(self, items):
        """检查列表中的所有元素是否相同"""
        return all(x == items[0] for x in items) if items else False


    # 修改原有方法：添加置信度参数
    def add_templates(self, event_template, confidence=1.0, insert=True, relevant_templates=[]):

            # if "<*>" not in event_template:
            #     self.template_tree["$CONSTANT_TEMPLATE$"][event_template] = event_template
            #     continue
            # original_template = event_template
            # event_template = self._preprocess_template(event_template)
            #print("event template after preprocess: ", event_template)
        template_tokens = message_split(event_template)
        if not template_tokens or event_template == "<*>":
            return -1
        if insert or len(relevant_templates) == 0:
            id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            return id
        # print("relevant templates: ", relevant_templates)
        max_similarity = 0
        similar_template = None
        for rt in relevant_templates:
            splited_template1, splited_template2 = rt.split(), event_template.split()
            if len(splited_template1) != len(splited_template2):
                continue 
            similarity = lcs_similarity(splited_template1, splited_template2)
            if similarity > max_similarity:
                max_similarity = similarity
                similar_template = rt
        if max_similarity > 0.8:
            success, id = self.modify(similar_template, event_template)
            if not success:
                id = self.insert(event_template, template_tokens, len(self.template_list))
                self.template_list.append(event_template)
            # 在插入后存储置信度
            self.store_confidence(id, confidence)
            return id
        else:
            id = self.insert(event_template, template_tokens, len(self.template_list))
            self.template_list.append(event_template)
            # 在插入后存储置信度
            self.store_confidence(id, confidence)
            return id
            #print("template tokens: ", template_tokens)
            
    def insert(self, event_template, template_tokens, template_id):
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

        move_tree["".join(template_tokens)] = (
            sum(1 for s in template_tokens if s != "<*>"),
            template_tokens.count("<*>"),
            event_template,
            template_id
        )  # statistic length, count of <*>, original_log, template_id
        return template_id

    def modify(self, similar_template, event_template):
        merged_template = []
        similar_tokens = similar_template.split()
        event_tokens = event_template.split()
        i = 0
        print(similar_template)
        print(event_template)
        for token in similar_tokens:
            print(token, event_tokens[i])
            if token == event_tokens[i]:
                merged_template.append(token)
            else:
                merged_template.append("<*>")
            i += 1
        merged_template = " ".join(merged_template)
        print("merged template: ", merged_template)
        success, old_ids = self.delete(similar_template)
        if not success:
            return False, -1
        self.insert(merged_template, message_split(merged_template), old_ids)
        self.template_list[old_ids] = merged_template
        return True, old_ids
        
    
    def delete(self, event_template):
        template_tokens = message_split(event_template)
        start_token = template_tokens[0]
        if start_token not in self.template_tree:
            return False, []
        move_tree = self.template_tree[start_token]

        tidx = 1
        while tidx < len(template_tokens):
            token = template_tokens[tidx]
            if token not in move_tree:
                return False, []
            move_tree = move_tree[token]
            tidx += 1
        old_id = move_tree["".join(template_tokens)][3]
        del move_tree["".join(template_tokens)]
        return True, old_id


    def match_event(self, log):
        return tree_match(self.template_tree, log)


    def _preprocess_template(self, template):
        # template = re.sub("<NUM>", "<*>", template)
        # if template.count("<*>") > 50:
        #     first_start_pos = template.index("<*>")
        #     template = template[0 : first_start_pos + 3]
        return template

    # 添加普通结构相似度分组方法（用于NO_WEIGHTING模式）
    def structural_similarity_grouping(self, new_template, candidates, threshold=SIMILARITY_THRESHOLD):
        """计算普通结构相似度（不考虑置信度）"""
        candidate_group = []
        new_tokens = new_template.split()

        for cand_template, cand_conf in candidates:
            # 计算结构相似度
            cand_tokens = cand_template.split()
            struct_sim = self.lcs_similarity(new_tokens, cand_tokens)

            if struct_sim >= threshold:
                candidate_group.append(cand_template)
        return candidate_group

def post_process_tokens(tokens, punc):
    excluded_str = ['=', '|', '(', ')']
    for i in range(len(tokens)):
        if tokens[i].find("<*>") != -1:
            tokens[i] = "<*>"
        else:
            new_str = ""
            for s in tokens[i]:
                if (s not in punc and s != ' ') or s in excluded_str:
                    new_str += s
            tokens[i] = new_str
    return tokens


#splitter_regex = re.compile("(<\*>|[^A-Za-z])")
def message_split(message):
    #print(string.punctuation)
    punc = "!\"#$%&'()+,-/:;=?@.[\]^_`{|}~"
    #print(punc)
    #punc = re.sub("[*<>\.\-\/\\]", "", string.punctuation)
    splitters = "\s\\" + "\\".join(punc)
    #print(splitters)
    #splitters = "\\".join(punc)
    # splitter_regex = re.compile("([{}]+)".format(splitters))
    splitter_regex = re.compile("([{}])".format(splitters))
    tokens = re.split(splitter_regex, message)

    tokens = list(filter(lambda x: x != "", tokens))
    
    #print("tokens: ", tokens)
    tokens = post_process_tokens(tokens, punc)

    tokens = [
        token.strip()
        for token in tokens
        if token != "" and token != ' ' 
    ]
    tokens = [
        token
        for idx, token in enumerate(tokens)
        if not (token == "<*>" and idx > 0 and tokens[idx - 1] == "<*>")
    ]
    #print("tokens: ", tokens)
    #tokens = [token.strip() for token in message.split()]
    #print(tokens)
    return tokens



def tree_match(match_tree, log_content):

    log_tokens = message_split(log_content)
        #print("log tokens: ", log_tokens)
    template, template_id, parameter_str = match_template(match_tree, log_tokens)
    if template:
        return (template, template_id, parameter_str)
    else:
        return ("NoMatch", "NoMatch", parameter_str)


def match_template(match_tree, log_tokens):
    results = []
    find_results = find_template(match_tree, log_tokens, results, [], 1)
    relevant_templates = find_results[1]
    if len(results) > 1:
        new_results = []
        for result in results:
            if result[0] is not None and result[1] is not None and result[2] is not None:
                new_results.append(result)
    else:
        new_results = results
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
            result = result + get_all_templates(value)
    return result


def find_template(move_tree, log_tokens, result, parameter_list, depth):
    flag = 0 # no futher find
    if len(log_tokens) == 0:
        for key, value in move_tree.items():
            if isinstance(value, tuple):
                result.append((key, value, tuple(parameter_list)))
                flag = 2 # match
        if "<*>" in move_tree:
            parameter_list.append("")
            move_tree = move_tree["<*>"]
            if isinstance(move_tree, tuple):
                result.append(("<*>", None, None))
                flag = 2 # match
            else:
                for key, value in move_tree.items():
                    if isinstance(value, tuple):
                        result.append((key, value, tuple(parameter_list)))
                        flag = 2 # match
        # return (True, [])
    else:
        token = log_tokens[0]

        relevant_templates = []
        
        if token in move_tree:
            find_result = find_template(move_tree[token], log_tokens[1:], result, parameter_list,depth+1)
            if find_result[0]:
                flag = 2 # match
            elif flag != 2:
                flag = 1 # futher find but no match
                relevant_templates = relevant_templates + find_result[1]
        if "<*>" in move_tree:
            if isinstance(move_tree["<*>"], dict):
                next_keys = move_tree["<*>"].keys()
                next_continue_keys = []
                for nk in next_keys:
                    nv = move_tree["<*>"][nk]
                    if not isinstance(nv, tuple):
                        next_continue_keys.append(nk)
                idx = 0
                # print("len : ", len(log_tokens))
                while idx < len(log_tokens):
                    token = log_tokens[idx]
                    # print("try", token)
                    if token in next_continue_keys:
                        # print("add", "".join(log_tokens[0:idx]))
                        parameter_list.append("".join(log_tokens[0:idx]))
                        # print("End at", idx, parameter_list)
                        find_result = find_template(
                            move_tree["<*>"], log_tokens[idx:], result, parameter_list,depth+1
                        )
                        if find_result[0]:
                            flag = 2 # match
                        elif flag != 2:
                            flag = 1 # futher find but no match
                            # relevant_templates = relevant_templates + find_result[1]
                        if parameter_list:
                            parameter_list.pop()
                    idx += 1
                if idx == len(log_tokens):
                    parameter_list.append("".join(log_tokens[0:idx]))
                    find_result = find_template(
                        move_tree["<*>"], log_tokens[idx + 1 :], result, parameter_list,depth+1
                    )
                    if find_result[0]:
                        flag = 2 # match
                    else:
                        if flag != 2:
                            flag = 1
                        relevant_templates = relevant_templates + find_result[1]
                    if parameter_list:
                        parameter_list.pop()
    if flag == 2:
        return (True, [])
    if flag == 1:
        return (False, relevant_templates)
    if flag == 0:
        # print(log_tokens, flag)
        if depth >= 2:
            return (False, get_all_templates(move_tree))
        else:
            return (False, [])