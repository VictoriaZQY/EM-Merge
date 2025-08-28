import sys

import regex as re
import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
from .gpt_query import query_template_from_gpt_with_check
from .parsing_cache import ParsingCache
from .prompt_select import prompt_select
from .post_process import correct_single_template
from .utils import load_pickle, save_pickle, load_tuple_list, cache_to_file, read_json_file
from tqdm import tqdm
# 在文件顶部添加导入
import numpy as np
from .parsing_cache import CONFIDENCE_THRESHOLD, SEMANTIC_THRESHOLD, SIMILARITY_THRESHOLD


def save_results_to_csv(log_file, template_file, cache_file, output_file, output_template_file):
    with open(log_file, 'r') as f:
        lines_a = f.readlines()
    with open(template_file, 'r') as f:
        lines_b = f.readlines()
    cache = load_pickle(cache_file)

    # 获取最大索引
    max_idx = max([int(line.split(' ')[0]) for line in lines_a]) if lines_a else 0
    total_expected = max_idx + 1

    # 创建完整的结果数组
    contents = [''] * total_expected
    event_templates = ['NoMatch'] * total_expected
    eventids = ['E0'] * total_expected  # 默认事件ID

    # 填充解析结果
    for line in lines_a:
        parts = line.strip().split(' ', 1)
        if len(parts) < 2:
            continue
        idx, content = parts
        idx = int(idx)
        if idx < total_expected:
            contents[idx] = content

    # 填充模板
    for line in lines_b:
        parts = line.strip().split(' ', 1)
        if len(parts) < 2:
            continue
        idx, template_id = parts
        idx = int(idx)
        template_id = int(template_id)
        if idx < total_expected:
            try:
                event_templates[idx] = cache.template_list[template_id]
            except:
                event_templates[idx] = "InvalidTemplate"

    # 创建模板映射
    templates_set = []
    template_id_map = {}
    for template in set(event_templates):
        if template != "NoMatch" and template != "InvalidTemplate":
            templates_set.append(template)
            template_id_map[template] = f"E{len(templates_set)}"

    # 设置事件ID
    for idx, template in enumerate(event_templates):
        if template in template_id_map:
            eventids[idx] = template_id_map[template]

    # 创建LineId列
    lineids = list(range(1, total_expected + 1))
    # ============== 修改结束 ==============

    df = pd.DataFrame({
        'LineId': lineids,
        'EventId': eventids,
        'Content': contents,
        'EventTemplate': event_templates
    })
    df.to_csv(output_file, index=False)

    template_ids = [f"E{i + 1}" for i in range(len(templates_set))]
    df = pd.DataFrame({'EventId': template_ids, 'EventTemplate': templates_set})
    df.to_csv(output_template_file, index=False)


def load_regs():
    regs_common = []
    with open("../logparser/LILAC/common.json", "r") as fr:
        dic = json.load(fr)
    
    patterns = dic['COMMON']['regex']
    for pattern in patterns:
        regs_common.append(re.compile(pattern))
    return regs_common


def check_model_files():
    model_path = os.path.join(os.path.dirname(__file__), "models", "all-MiniLM-L6-v2")
    required_files = [
        "config.json", "pytorch_model.bin", "sentence_bert_config.json",
        "tokenizer.json", "vocab.txt", "special_tokens_map.json", "tokenizer_config.json"
    ]

    missing_files = []
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            missing_files.append(file)

    if missing_files:
        print(f"Error: Missing model files: {', '.join(missing_files)}")
        print("Please download the following files from:")
        print("https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/tree/main")
        print("and place them in the directory:", model_path)
        sys.exit(1)



class LogParser:
    check_model_files()

    def __init__(self, log_format, indir='./', outdir='./result/', rex=[],
                 data_type='2k', shot=0, example_size=0, model="gpt-3.5-turbo-0613", selection_method="LILAC"):
        self.path = indir
        self.df_log = None
        self.log_format = log_format
        self.data_type = data_type
        self.shot = shot
        self.example_size = example_size
        self.selection_method = selection_method
        self.model = model

    def parse(self, logName):
        print('Parsing file: ' + os.path.join(self.path, logName))
        start_time = datetime.now()
        self.logName = logName
        dataset_name = logName.split('_')[0]
        output_path = os.path.join(f"../../temp/lilac_temp_{self.data_type}_{self.shot}_{self.example_size}_{self.model}", dataset_name)
        evaluation_path = f"../../result/result_LILAC_{self.data_type}_{self.shot}_{self.example_size}_{self.model}/"
        if os.path.exists(os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_structured.csv")):
            print(f"{dataset_name} already exists.")
            return
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        if not os.path.exists(evaluation_path):
            os.makedirs(evaluation_path)
        self.load_data()

        regs_common = load_regs()

        cached_tree = os.path.join(output_path, "cached_tree.pkl")
        cached_log = os.path.join(output_path, "cached_log.txt")
        cached_template = os.path.join(output_path, "cached_template.txt")
        if os.path.exists(cached_tree) and os.path.exists(cached_log) and os.path.exists(cached_template):
            cache = load_pickle(cached_tree)
            log_messages = load_tuple_list(cached_log)
            log_templates = load_tuple_list(cached_template)
            idx = log_messages[-1][1]
        else:
            log_messages = []
            log_templates = []
            cache = ParsingCache()
            idx = 0

        prompt_cases = None if self.shot == 0 else read_json_file(f"../../full_dataset/sampled_examples/{dataset_name}/{self.shot}shot.json")

        num_query = 0
        total_line = len(self.df_log)
        cache_step = total_line // 5

        if idx + 1 < total_line:
            for i in range(idx, total_line):
                log = self.df_log.iloc[i]['Content']
                flag = self.process_log(cache, [log], log_messages, log_templates, i, prompt_cases, regs_common,
                                        total_line)
                if flag:
                    num_query += 1
                    print("Query times: ", num_query)
                if i % cache_step == 0:
                    print("Finished processing line: ", i)
                    cache_to_file(log_messages, cached_log)
                    cache_to_file(log_templates, cached_template)
                    save_pickle(cache, cached_tree)

            # 添加完整性检查
            processed_indices = set(msg[1] for msg in log_messages)
            if len(processed_indices) < total_line:
                missing_count = total_line - len(processed_indices)
                print(f"警告: {missing_count} 行日志未被处理")

                # 填充缺失行
                for i in range(total_line):
                    if i not in processed_indices:
                        log = self.df_log.iloc[i]['Content']
                        log_messages.append((log, i))
                        log_templates.append((-1, i))  # 使用-1表示缺失
        if num_query > 0:
            print("Total query: ", num_query)

        cache_to_file(log_messages, cached_log)
        cache_to_file(log_templates, cached_template)
        save_pickle(cache, cached_tree)
        save_results_to_csv(cached_log, cached_template, cached_tree,
                            os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_structured.csv"),
                            os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_templates.csv"))

        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def process_log(self, cache, logs, log_messages, log_templates, idx, prompt_cases, regs_common, total_line):
        try:
            new_template = None
            log = logs[0]  # 每次只处理一行日志
            results = cache.match_event(log)

            if results[0] == "NoMatch":
                print("===========================================")
                print(f"Line-{idx}/{total_line}: No match. {log}")
                if prompt_cases != None:
                    examples = prompt_select(prompt_cases, log, self.example_size, self.selection_method)
                else:
                    examples = []

                # 获取置信度
                new_template, normal, confidence = query_template_from_gpt_with_check(
                    log, regs_common, examples, self.model
                )
                print("queried_new_template: ", new_template)

                # 新增：EM-Merge后处理模块
                if confidence > CONFIDENCE_THRESHOLD:  # 仅处理高置信度模板
                    # 语义嵌入近邻搜索
                    semantic_neighbors = cache.find_semantic_neighbors(
                        new_template,
                        SEMANTIC_THRESHOLD
                    )

                    if semantic_neighbors:
                        # 置信度加权结构相似度计算
                        candidate_group = cache.weighted_similarity_grouping(
                            new_template,
                            confidence,
                            semantic_neighbors,
                            SIMILARITY_THRESHOLD
                        )

                        if candidate_group:
                            # 模板合并
                            merged_template = cache.merge_templates(
                                [new_template] + candidate_group
                            )
                            print(f"Merged template: {merged_template} (from {len(candidate_group) + 1} templates)")
                            new_template = merged_template

                # 后处理模块结束

                template_id = cache.add_templates(new_template, confidence, normal, results[2])
                log_messages.append((log, idx))
                log_templates.append((template_id, idx))
                print("===========================================")
                return True
            else:
                log_messages.append((log, idx))
                log_templates.append((results[1], idx))
                return False
        except Exception as e:
            print(f"处理日志行 {idx} 时出错: {e}")
            # 出错时使用默认值
            log = logs[0]
            new_template = "NoMatch"
            template_id = cache.add_templates("NoMatch", 0, True, [])
            log_messages.append((log, idx))
            log_templates.append((template_id, idx))
            print("===========================================")
            return True


    def load_data(self):
        csv_path = os.path.join(self.path, self.logName+'_structured.csv')
        if os.path.exists(csv_path):
            self.df_log = pd.read_csv(csv_path)
        else:
            headers, regex = self.generate_logformat_regex(self.log_format)
            self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)


    def log_to_dataframe(self, log_file, regex, headers, logformat):
        """ Function to transform log file to dataframe 
        """
        log_messages = []
        linecount = 0
        with open(log_file, 'r') as fin:
            for line in fin.readlines():
                try:
                    match = regex.search(line.strip())
                    message = [match.group(header) for header in headers]
                    log_messages.append(message)
                    linecount += 1
                except Exception as e:
                    pass
        logdf = pd.DataFrame(log_messages, columns=headers)
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf


    def generate_logformat_regex(self, logformat):
        """ Function to generate regular expression to split log messages
        """
        headers = []
        splitters = re.split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = re.sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
                headers.append(header)
        regex = re.compile('^' + regex + '$')
        return headers, regex
