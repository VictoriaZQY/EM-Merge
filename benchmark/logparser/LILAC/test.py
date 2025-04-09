import math
import os
import pandas as pd
from datetime import datetime
from .gpt_query import query_template_from_gpt_with_check
from .prompt_select import prompt_select
from .post_process import post_average
from .utils import load_pickle, save_pickle, load_tuple_list, cache_to_file, read_json_file
from merge_template import get_similar_templates, merge_templates_by_lcs
from split_template import split_template_by_lcs

# 改进版缓存类，参考 YALP 中 DataHandler 的相关逻辑
class ParsingCache:
    def __init__(self):
        # 模板列表，用于存储所有解析出的模板（字符串形式）
        self.template_list = []  # 模板列表，索引即模板ID（0开始）
        # 样本列表，每个样本以字典形式存储：{'log': 日志文本, 'template_id': 模板索引, 'idx': 日志行号}
        self.samples = []

    def match_event(self, log: str):
        """
        遍历已有模板，检查当前日志是否能够匹配其中某个模板。
        这里简单采用字符串包含判断（实际场景可替换为正则匹配或更复杂的相似度计算）。
        若匹配成功，返回 ("Match", template_id, extra_info)；否则返回 ("NoMatch", None, None)。
        """
        for idx, template in enumerate(self.template_list):
            # 跳过已被删除的模板
            if template == "<REMOVED>":
                continue
            # 简单判断：如果模板字符串出现在日志中，则认为匹配成功
            if template in log:
                return ("Match", idx, None)
        return ("NoMatch", None, None)

    def add_templates(self, new_template: str, normal, extra_info) -> int:
        """
        添加新模板。如果模板已存在，则直接返回对应模板ID；否则添加到模板列表中，
        并返回新模板的索引（ID）。
        """
        if new_template in self.template_list:
            return self.template_list.index(new_template)
        else:
            self.template_list.append(new_template)
            return len(self.template_list) - 1

    def get_samples_by_template(self, template_id: int) -> list:
        """
        根据模板ID返回所有对应的样本。模拟 YALP 中 select_samples_by_template_id 方法。
        """
        return [sample for sample in self.samples if sample['template_id'] == template_id]

    def update_sample_template(self, sample: dict, new_template_id: int) -> None:
        """
        更新样本中记录的模板ID。模拟 YALP 中 update_template_id_of_sample 方法。
        """
        sample['template_id'] = new_template_id

    def remove_template(self, template_id: int) -> None:
        """
        将指定模板标记为删除。这里直接将模板内容置为 "<REMOVED>"，
        后续新样本不再匹配到该模板。
        """
        if 0 <= template_id < len(self.template_list):
            self.template_list[template_id] = "<REMOVED>"

    def add_sample(self, log: str, template_id: int, idx: int) -> None:
        """
        添加一个新的样本记录到缓存中。
        """
        sample = {'log': log, 'template_id': template_id, 'idx': idx, 'log_num': 1}
        self.samples.append(sample)

    def update_sample_log(self, sample: dict) -> None:
        """
        更新样本日志计数（类似于 YALP 中 update_log_num_of_sample）。
        """
        sample['log_num'] += 1

    def select_sample_len_by_template(self, template_id: int) -> int:
        """
        计算指定模板对应的样本数量的总和，模拟 YALP 中 select_sample_len_by_template_id。
        """
        samples = self.get_samples_by_template(template_id)
        return sum(sample.get('log_num', 1) for sample in samples)


# 改进版 LogParser 类，引入了模板合并和拆分的逻辑
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
        self.split_threshold = split_threshold  # 模板拆分阈值：当对应样本总数超过此值时触发拆分

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
                # 对于每条日志，无论匹配与否，都记录为一个样本；若样本已存在则更新计数
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
        逐行处理日志：
         - 首先在缓存中查找是否已有匹配模板；
         - 若无匹配，则调用 GPT 查询生成新模板，并尝试与已有模板合并，然后更新缓存；
         - 更新样本记录后检查模板下样本数量是否超过阈值，若是则触发模板拆分。
        """
        results = cache.match_event(log)
        if results[0] == "NoMatch":
            print("===========================================")
            print(f"Line-{idx}/{total_line}: No match. {log}")
            if prompt_cases is not None:
                examples = prompt_select(prompt_cases, log, self.example_size, self.selection_method)
            else:
                examples = []
            # 调用 GPT 生成新模板
            new_template, normal = query_template_from_gpt_with_check(log, regs_common, examples, self.model)
            # 尝试与缓存中已有模板合并
            new_template = self.template_merge(new_template, cache)
            print("Merged/Generated Template: ", new_template)
            # 将新模板加入缓存，返回模板 id
            template_id = cache.add_templates(new_template, normal, results[2])
            log_messages.append((log, idx))
            log_templates.append((template_id, idx))
            # 添加样本记录到缓存
            cache.add_sample(log, template_id, idx)
            # 检查是否需要拆分模板（若样本总数超过阈值）
            self.template_split(template_id, cache)
            print("===========================================")
            return True
        else:
            template_id = results[1]
            log_messages.append((log, idx))
            log_templates.append((template_id, idx))
            # 添加或更新样本记录
            # 先查找是否已有该行日志记录（简单判断），否则添加为新样本
            samples = cache.get_samples_by_template(template_id)
            found = False
            for sample in samples:
                if sample['log'] == log:
                    cache.update_sample_log(sample)
                    found = True
                    break
            if not found:
                cache.add_sample(log, template_id, idx)
            # 同样检查该模板是否需要拆分
            self.template_split(template_id, cache)
            return False

    def template_merge(self, new_template: str, cache) -> str:
        """
        模板合并：
         - 从缓存中获取已有模板列表；
         - 利用外部函数 get_similar_templates 找到与新模板相似的模板；
         - 若存在相似模板，则调用 merge_templates_by_lcs 进行合并，返回合并后的模板，否则返回原模板。
        """
        existing_templates = [tpl for tpl in cache.template_list if tpl != "<REMOVED>"]
        similar_templates = get_similar_templates(existing_templates, new_template, cluster_enable=True)
        if similar_templates:
            merged_template = merge_templates_by_lcs(similar_templates)
            if merged_template and len(merged_template.strip()) > 0:
                return merged_template
        return new_template

    def template_split(self, template_id, cache) -> None:
        """
        模板拆分：
         - 获取缓存中与 template_id 对应的所有样本；
         - 如果样本数量超过预设阈值，则利用 split_template_by_lcs 进行模板拆分；
         - 移除原模板，并将拆分出的新模板添加到缓存，同时更新相应样本的模板ID。
        """
        sample_count = cache.select_sample_len_by_template(template_id)
        if sample_count >= self.split_threshold:
            samples = cache.get_samples_by_template(template_id)
            # 根据样本数量计算拆分因子，例如使用对数函数
            split_factor = round(math.log(sample_count, 2))
            split_results = split_template_by_lcs(samples, split_factor)
            # 移除原模板
            cache.remove_template(template_id)
            for new_template_text, split_samples in split_results:
                new_id = cache.add_templates(new_template_text, normal=True, extra_info=None)
                for sample in split_samples:
                    cache.update_sample_template(sample, new_id)
            print(f"Template {template_id} split into {len(split_results)} templates due to high sample count ({sample_count}).")

    def load_data(self):
        csv_path = os.path.join(self.path, self.logName+'_structured.csv')
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
        logdf.insert(0, 'LineId', None)
        logdf['LineId'] = [i + 1 for i in range(linecount)]
        return logdf

    def generate_logformat_regex(self, logformat):
        headers = []
        splitters = __import__('re').split(r'(<[^<>]+>)', logformat)
        regex = ''
        for k in range(len(splitters)):
            if k % 2 == 0:
                splitter = __import__('re').sub(' +', '\\\s+', splitters[k])
                regex += splitter
            else:
                header = splitters[k].strip('<').strip('>')
                regex += '(?P<%s>.*?)' % header
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
        lineids = range(1, total_len + 1)
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
