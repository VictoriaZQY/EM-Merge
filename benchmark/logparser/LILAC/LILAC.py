import math

import os
import sys

from .Template import Template

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from datetime import datetime
from .gpt_query import query_template_from_gpt_with_check
from .parsing_cache import ParsingCache
from .prompt_select import prompt_select
from .utils import load_pickle, save_pickle, load_tuple_list, cache_to_file, read_json_file
from merge_template import get_similar_templates, merge_templates_by_lcs
from split_template import split_template_by_lcs
from .Template import Template  # 确保引入 Template 类

###########################################################
# 改进版 LogParser 类
###########################################################
class LogParser:
    def __init__(self, log_format, indir='./', outdir='./result/', rex=[],
                 data_type='2k', shot=0, example_size=0, model="gpt-3.5-turbo-0613", selection_method="LILAC",
                 split_threshold: int = 50):
        self.path = indir
        self.df_log = None
        self.log_format = log_format
        self.data_type = data_type
        self.shot = shot
        self.example_size = example_size
        self.selection_method = selection_method
        self.model = model
        self.split_threshold = split_threshold  # 当模板下样本总数超过该阈值时触发拆分

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
        regs_common = self.load_regs()

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
        cache_step = max(total_line // 5, 1)
        if idx + 1 < total_line:
            for log in list(self.df_log[idx:]['Content']):
                flag = self.process_log(cache, log, log_messages, log_templates, idx, prompt_cases, regs_common, total_line)
                if flag:
                    num_query += 1
                idx += 1
                if idx % cache_step == 0:
                    print("Finished processing line: ", idx)
                    cache_to_file(log_messages, cached_log)
                    cache_to_file(log_templates, cached_template)
                    save_pickle(cache, cached_tree)
        if num_query > 0:
            print("Total query: ", num_query)
        cache_to_file(log_messages, cached_log)
        cache_to_file(log_templates, cached_template)
        save_pickle(cache, cached_tree)
        self.save_results_to_csv(cached_log, cached_template, cached_tree,
                                 os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_structured.csv"),
                                 os.path.join(evaluation_path, f"{dataset_name}_{self.data_type}.log_templates.csv"))
        print('Parsing done. [Time taken: {!s}]'.format(datetime.now() - start_time))

    def process_log(self, cache, log, log_messages, log_templates, idx, prompt_cases, regs_common, total_line):
        """
        对每一行日志：
         - 先检查缓存中是否有匹配模板；
         - 若无匹配，则调用 GPT 生成模板，并尝试合并；
         - 将日志样本记录加入缓存，并检查是否需要触发模板拆分。
        """
        results = cache.match_event(log)
        if results[0] == "NoMatch":
            print("===========================================")
            print(f"Line-{idx}/{total_line}: No match. {log}")
            examples = prompt_select(prompt_cases, log, self.example_size, self.selection_method) if prompt_cases else []
            new_template, normal = query_template_from_gpt_with_check(log, regs_common, examples, self.model)
            # 尝试与缓存中已有模板合并
            new_template = self.template_merge(new_template, cache)
            print("Merged/Generated Template: ", new_template)
            # 将新模板加入缓存，返回模板 id
            template_id = cache.add_templates(new_template, normal, results[2])
            log_messages.append((log, idx))
            log_templates.append((template_id, idx))
            cache.add_sample_to_template(log, template_id, idx)
            self.template_split(template_id, cache)
            print("===========================================")
            return True
        else:
            template_id = results[1]
            log_messages.append((log, idx))
            log_templates.append((template_id, idx))
            # 若日志已存在，则更新计数；否则新增样本记录
            samples = cache.get_samples_by_template(template_id)
            found = False
            for sample in samples:
                if sample['log'] == log:
                    cache.update_sample_log(sample)
                    found = True
                    break
            if not found:
                cache.add_sample_to_template(log, template_id, idx)
            self.template_split(template_id, cache)
            return False

    def template_merge(self, new_template, cache) -> str:
        """
        模板合并：
         - 将 new_template 包装为 Template 对象（如果尚未包装）；
         - 从缓存中获取所有已有模板；
         - 利用 get_similar_templates 找出与新模板相似的模板；
         - 若存在相似模板，则调用 merge_templates_by_lcs 进行合并，返回合并后的模板文本，否则返回原模板文本。
        """
        from .Template import Template  # 确保引入 Template 类
        if not hasattr(new_template, "template_text"):
            new_template_obj = Template(new_template)
        else:
            new_template_obj = new_template

        existing_templates = []
        for tpl in cache.template_list:
            if hasattr(tpl, "template_text"):
                existing_templates.append(tpl)
            else:
                existing_templates.append(Template(tpl))

        similar_templates = get_similar_templates(existing_templates, new_template_obj, cluster_enable=True)
        if similar_templates:
            merged_template = merge_templates_by_lcs(similar_templates)
            # 修改处：使用 merged_template.template_text.strip() 而非 merged_template.strip()
            if merged_template and merged_template.template_text.strip():
                return merged_template.template_text
        return new_template_obj.template_text

    def template_split(self, template_id, cache) -> None:
        """
        检查模板下的样本总数，若超过拆分阈值则触发模板拆分操作：
         - 根据样本数量计算拆分因子，利用外部函数 split_template_by_lcs 进行拆分；
         - 删除原模板，并为每个拆分结果插入新模板，同时更新对应样本记录。
        """
        sample_count = cache.select_sample_len_by_template(template_id)
        if sample_count >= self.split_threshold:
            samples = cache.get_samples_by_template(template_id)
            split_factor = round(math.log(sample_count, 2))
            split_results = split_template_by_lcs(samples, split_factor)
            cache.remove_template(template_id)
            for new_template_text, split_samples in split_results:
                new_id = cache.add_templates(new_template_text, normal=True, extra_info=[])
                for sample in split_samples:
                    cache.update_sample_template(sample, new_id)
            print(f"Template {template_id} split into {len(split_results)} templates (sample count: {sample_count}).")

    def load_data(self):
        csv_path = os.path.join(self.path, self.logName + '_structured.csv')
        if os.path.exists(csv_path):
            self.df_log = pd.read_csv(csv_path)
        else:
            headers, regex = self.generate_logformat_regex(self.log_format)
            self.df_log = self.log_to_dataframe(os.path.join(self.path, self.logName), regex, headers, self.log_format)

    def log_to_dataframe(self, log_file, regex, headers, logformat):
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
        logdf.insert(0, 'LineId', list(range(1, linecount + 1)))
        return logdf

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = __import__('re').split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k, part in enumerate(splitters):
            if k % 2 == 0:
                splitter = __import__('re').sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = part.strip('<>').strip()
                regex += f'(?P<{header}>.*?)'
                headers.append(header)
        regex = __import__('re').compile('^' + regex + '$')
        return headers, regex

    def load_regs(self):
        regs_common = []
        with open("../logparser/LILAC/common.json", "r") as fr:
            dic = __import__('json').load(fr)
        patterns = dic['COMMON']['regex']
        for pattern in patterns:
            regs_common.append(__import__('re').compile(pattern))
        return regs_common

    def save_results_to_csv(self, log_file, template_file, cache_file, output_file, output_template_file):
        with open(log_file, 'r') as f:
            lines_a = f.readlines()
        with open(template_file, 'r') as f:
            lines_b = f.readlines()
        cache = load_pickle(cache_file)
        total_len = len(lines_a)
        lineids = list(range(1, total_len + 1))
        eventids = [''] * total_len
        contents = [''] * total_len
        event_templates = [''] * total_len
        templates_set = []
        print("start writing log structured csv.")
        for (line_a, line_b) in zip(lines_a, lines_b):
            idx_a, str_a = line_a.strip().split(' ', 1)
            idx_b, str_b = line_b.strip().split(' ', 1)
            idx_a, idx_b = int(idx_a), int(idx_b)
            str_b = cache.template_list[int(str_b)]
            if idx_a != idx_b:
                print(f"Error in line: {idx_a} {idx_b}")
                return
            if str_b in templates_set:
                template_id = templates_set.index(str_b) + 1
            else:
                templates_set.append(str_b)
                template_id = len(templates_set)
            contents[idx_a] = str_a
            event_templates[idx_a] = str_b
            eventids[idx_a] = f"E{template_id}"
        print("end writing log structured csv.")
        df = pd.DataFrame({'LineId': lineids, 'EventId': eventids, 'Content': contents, 'EventTemplate': event_templates})
        df.to_csv(output_file, index=False)
        template_ids = [f"E{i+1}" for i in range(len(templates_set))]
        df = pd.DataFrame({'EventId': template_ids, 'EventTemplate': templates_set})
        df.to_csv(output_template_file, index=False)
