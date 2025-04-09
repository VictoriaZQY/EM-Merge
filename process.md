# LILAC_eval.py
start from the main.

该脚本的主要目的是通过解析日志文件并评估不同的日志模板，计算多个准确性指标（如分组准确性、解析准确性等）。根据命令行参数，脚本会选择适当的数据集，评估每个数据集，并将结果保存在指定的文件中。最后，通过后处理计算并保存结果的平均值。

## Workflow:
1. **Input Arguments**: User specifies settings such as data size (`full` or `2k`), model parameters, and evaluation options.
2. **Directory Setup**: Paths are configured for input and output.
3. **Dataset Processing**: For each dataset, the logs are parsed and evaluated.
4. **Postprocessing**: Metrics are averaged and saved.


## LILAC_eval.py, 64
### common.py
#### 87 parser = argparse.ArgumentParser()
class ArgumentParser(_AttributeHolder, _ActionsContainer):

Object for parsing command line strings into Python objects.

    Keyword Arguments:
        - prog -- The name of the program (default: sys.argv[0])  -- 'LILAC_eval.py'
        - usage -- A usage message (default: auto-generated from arguments)
        - description -- A description of what the program does
        - epilog -- Text following the argument descriptions
        - parents -- Parsers whose arguments should be copied into this one
        - formatter_class -- HelpFormatter class for printing help messages
        - prefix_chars -- Characters that prefix optional arguments
        - fromfile_prefix_chars -- Characters that prefix files containing
            additional arguments
        - argument_default -- The default value for all arguments
        - conflict_handler -- String indicating how to handle conflicts -- 'error'
        - add_help -- Add a -h/-help option  -- True
        - allow_abbrev -- Allow long options to be abbreviated unambiguously
#### 88-108: add arguments and parse back to arguments.
args: Namespace(complex=0, example_size=0, frequent-0, full_data=True, model='gpt-3.5-turbo-0613', oracle_template_correction=False, selection_method='LILAC', shot=0)

## 65-68: data_type: 'full', 找到output的格式和要求

## 71 进入evaluation
调用 prepare_results() 函数准备评估结果文件，返回结果文件的文件名 result_file。该函数根据 otc、complex 和 frequent 等参数生成 CSV 文件，并创建目录结构。

### 转向evaluator_main.py, 32

#### prepare_results 函数
- 检查输出目录 output_dir 是否存在，如果不存在则创建它。
- 创建一个新的CSV文件 result_file，并在文件中写入列标题。列标题包括：

  - Dataset: 数据集名称
  - parse_time: 解析时间 
  - identified_templates: 识别到的模板数量 
  - ground_templates: 真实模板数量 
  - 其他评价指标如 GA (Grouping Accuracy)、PA (Parsing Accuracy) 等。
- 返回创建的结果文件的文件名。

## 转回LILAC_eval 71，with 创建的文件，78判断数据集，82逐个进行
- 遍历选定的数据集 datasets，为每个数据集执行评估。 
- 从 benchmark_settings 中获取该数据集的设置（包括日志文件等信息）。 
- 使用 replace 方法调整日志文件的名称，将 _2k 替换为数据类型（full 或 2k） 
- 检查输出目录中是否已存在解析后的日志文件。如果已存在文件，则不需要再次解析，设置 parser = None。 
- 如果文件不存在，则使用 LogParser 类解析日志。
## 92：调用 evaluator() 函数，开始对每个数据集进行评估。
  - dataset：当前数据集名称。 
  - input_dir、output_dir：输入和输出目录。 
  - log_file：日志文件路径。 
  - LogParser：解析器类（或 None）。 
  - param_dict：包含多个参数的字典，如日志格式、数据类型、模型、样本数量等。 
  - otc、complex、frequent：控制是否使用纠正模板、模板复杂度和模板频率的参数。 
  - result_file：结果文件路径。
### 转evaluator_main，73
这个函数负责对特定的数据集进行评估。它会处理日志文件，解析日志，计算准确率等指标，并将结果保存到CSV文件中。
- 74，75：构建日志文件所在的目录 indir 和日志文件的基本文件名 log_file_basename。
- 根据 otc 参数，决定使用纠正过的oracle模板 (_structured_corrected.csv) 还是原始的结构化日志文件 (_structured.csv) 作为地面真值。
- 设置解析后的日志结果文件路径。
- 记录开始时间，用于后续计算解析时间。

### 85
如果 LogParser 不为 None，则启动一个新的进程来解析日志文件。如果解析时间超过了 TIMEOUT（48小时），则终止进程并记录超时。否则，记录解析时间。

#### 90 转LILAC.py，66：LogParser
parse 函数：

80-94：
- 开始解析指定的日志文件 logName。 
- 设置输入输出路径，并根据文件名创建缓存目录。 
- 如果解析结果已经存在，则跳过解析过程。
- 加载数据和正则表达式规则。

96-108：
- 检查是否存在缓存文件，若存在则加载缓存；否则，创建新的缓存和日志信息列表。 

110：
- 根据 shot 参数决定是否加载用于 GPT 查询的示例数据。 

112-126：
- 遍历日志行，并处理每一行日志。 
- 调用 process_log 函数进行日志处理，若需要查询 GPT，则统计查询次数。 
- 定期保存处理进度和缓存。 

127-137：
- 处理完所有日志行后，保存处理结果、缓存和生成的 CSV 文件。


### 107
检查解析结果文件是否存在且不为空。如果文件不存在或为空，记录一行表示该数据集没有生成结果，并返回。

### 127
读取解析结果文件和地面真值文件，并将 NaN 填充为空字符串。

### 133
初始化模板过滤器。

### 134-144
根据 complex 参数过滤模板。模板按其包含的 "<*>"（表示通配符的占位符）的数量进行过滤：
- complex == 1：仅保留不包含通配符的模板。
- complex == 2：保留包含1到4个通配符的模板。
- complex == 3：保留包含5个及以上通配符的模板。

### 156-166
根据 frequent 参数过滤模板。模板按出现频率排序，然后选择出现次数最多的 n% 或最少的 n% 的模板。

### 171-185
如果过滤后的模板数量为0，记录结果为“None”并返回。

### 188-194
打印消息并开始计算分组准确率（GA）。使用 evaluate 函数计算分组准确率，GA 和 FGA。

#### 191 转evaluator.py 35
- 功能：这是主评估函数，用于计算日志解析准确性。 
- 参数： 
  - df_groundtruth：包含地面真值的DataFrame（即真实的日志模板）。
  - df_parsedlog：包含解析结果的DataFrame。 
  - filter_templates：可选参数，用于过滤特定的模板。 
- 返回值： 返回分组准确率（GA）和过滤分组准确率（FGA）。

#### 35-37
过滤掉在 df_groundtruth 中 EventTemplate 为 NaN 的记录（无效的日志事件）。

同样地，df_parsedlog 中的记录也根据 df_groundtruth 中有效的事件进行同步筛选。

#### 38-40
调用 get_accuracy 函数计算分组准确率（GA）和过滤分组准确率（FGA）。
返回 GA 和 FGA。

#### 61 get_accuracy
- 功能：计算日志解析结果与地面真值之间的准确度指标。 
- 参数：
  - series_groundtruth：地面真值中的事件模板ID。 
  - series_parsedlog：解析后的事件模板ID。 
  - filter_templates：可选参数，用于筛选特定的模板。
- 返回 GA、PGA、RGA 和 FGA 四个指标。

#### 61-64
获取地面真值和解析结果的模板事件ID计数。

将 series_groundtruth 和 series_parsedlog 合并成一个DataFrame，并按 groundtruth 对数据进行分组。

#### 65-68
初始化 accurate_events 和 accurate_templates，分别用于记录解析正确的事件数量和模板数量。

如果提供了 filter_templates，则初始化一个集合 filter_identify_templates 用于记录被识别的过滤模板。 (未进入)

#### 70-72
使用 tqdm 显示进度条，遍历每个地面真值ID（ground_truthId）。

对于每个组（即相同地面真值ID的所有日志事件），计算解析结果的事件ID计数。

#### 73-75
如果提供了 filter_templates 且当前的地面真值ID在 filter_templates 中，则将识别的解析事件ID添加到 filter_identify_templates 集合中。

#### 76-81
如果解析结果中只有一个事件ID，且该事件ID的大小等于地面真值组的大小，则认为解析成功，增加正确事件和正确模板的计数。

如果提供了 filter_templates，仅在该地面真值ID属于过滤模板时才进行计数。

#### 86-93
根据 filter_templates 是否为 None 计算不同的准确率：
- GA：分组准确率。
- PGA：精确度。
- RGA：召回率。

如果没有 filter_templates，则计算所有地面真值和解析结果的总体准确率。

##### 88 转LILAC.py, parse
- 功能: 解析指定的日志文件。
- 解析过程:
  - 根据输入的 logName，设置输出路径和评估路径。
  - 如果解析的日志已经存在，跳过解析。
  - 加载日志数据并准备缓存文件。
  - 通过缓存文件加载已解析的日志消息和模板，或者初始化新的解析缓存。
  - 读取日志文件并将每一条日志进行处理。如果有匹配不到的模板，则通过GPT生成新的模板，并将其加入缓存。

##### LILAC 117 转 LILAC 140 process_log
- 功能: 处理每一条日志。
- 解析过程:
  - 使用缓存（cache）匹配日志（log）。
  - 如果没有找到匹配的模板，则通过GPT生成新的模板，并将其添加到缓存中。
  - 将日志和对应的模板 ID 保存到 log_messages 和 log_templates 中。
  - 如果日志匹配成功，则直接返回匹配的模板 ID。

###### LILAC 143 转 parsing_cache.py 141 match_event
- 功能: 使用树结构来匹配日志内容。 
- 过程: 调用 tree_match 函数来匹配日志内容。

###### 202 tree_match (parsing_cache.py)
- 功能: 使用树结构匹配日志内容。 
- 过程:
  - 将日志内容分割成令牌。 
  - 调用 match_template 函数来在树中查找与日志匹配的模板。 
  - 返回匹配的模板、模板 ID 和参数字符串。

###### 168 message_split
- 功能: 将日志消息分割成令牌（tokens）。
- 过程:
  - 定义分隔符（包括空格和标点符号），并使用正则表达式将消息分割成令牌。 
  - 处理令牌中的标点符号，并将 <*> 作为特殊令牌保留。 
  - 去掉空白字符，并删除连续的 <*> 令牌。

###### 转 204 tree_match

###### 206 转 214-228 match_template
该函数用于在模板树中查找与日志 tokens 最匹配的模板。

过程：
- 调用 find_template() 来查找匹配的模板。
  - 该函数用于递归地在模板树中查找与日志 token 匹配的模板。 
  - 过程： 
    - 如果日志 token 用尽，则检查模板树中是否存在 <*>，并根据当前的参数列表返回匹配的模板。 
    - 如果日志 token 不为空，则检查当前 token 是否存在于模板树中： 
    - 如果存在，继续递归匹配下一个 token。 
    - 如果存在 <*>，则尝试匹配多个不同的子模板。 
    - 递归过程会收集所有匹配的模板并返回。 
  - 该函数是实现模板匹配的核心部分，通过递归遍历模板树，尝试匹配日志中的每个 token。
- 如果找到多个匹配的模板，则按某些标准进行排序（例如，根据模板的长度和泛化程度）。
- 返回最佳匹配的模板、其 ID 和参数字符串。

该函数的目的是根据日志的分词结果找到最合适的模板。

##### LILAC 144-151 

###### 转 gpt_query.py 150 query_template_from_gpt_with_check
This function combines querying the GPT model for a log template and post-processing the response to ensure the template is valid and matches the log message correctly.

Input:
- log_message: The log message to parse. 
- regs_common: A list of regular expressions for post-processing the template.
- examples: A list of example query-answer pairs to help fine-tune the model's responses.
- model: The OpenAI model to use for the request (default is 'gpt-3.5-turbo-0613').

Process:
- It first calls query_template_from_gpt to get an initial log template from the model.
- If the template is valid (not empty and can be processed correctly), it passes it through post_process_template to ensure it’s refined and valid.
- Then, it checks whether the template matches the original log message using a matching function (tree.match_event()).
- If the template matches, it returns the template. If it doesn't, it prints an error message.

###### 150转98 query_template_from_gpt：

- This function queries the OpenAI model to generate a log template based on a log message. 
- Calls get_response_from_openai_key to get a response from the OpenAI model, and processes the result to extract the log template. 
- Output: Returns the log template, if found.

###### 102转86 get_response_from_openai_key：
- This function provides a simplified interface for calling the infer_llm function and retrieving a parsed log template. 
- Process: It formats the input and uses the infer_llm function to get a response for the log parsing. 
- Output: Returns the parsed log template.

###### 93转21 infer_llm：
- This function interacts with the OpenAI API to process log parsing based on a provided model. 
- Process:
  - If using the gpt-3.5-turbo-0613 model, it formats the messages in a structured format suitable for a chat-based model (system, user, assistant roles). 
  - If using a different model, it creates a simpler text-based prompt. 
  - It tries to send the message to the OpenAI API up to 3 times, retrying in case of failures.
- Output: Returns the response from the OpenAI API for log parsing.

整体逻辑：
- 与OpenAI API交互：函数infer_llm是这个过程的核心，它构造了一组模拟对话（系统、用户、助手）的消息。它还包括以前日志解析交互的示例（范例），以指导模型的行为。
- 重试逻辑:如果请求失败（由于连接问题或“列表索引超出范围”之类的错误），函数将重试请求多达3次以确保健壮性。
- 模板生成:在模型生成日志模板之后，代码通过将占位符（如{}）替换为通用占位符<*>来处理它。post_process_template函数有助于确保生成的模板不会太宽泛，并且可以与真实的日志消息匹配。
使用tree.match_event（log_message）验证最终的日志模板，以确保它与提供的日志消息匹配。
- 错误处理:该代码包括对错误的几项检查，例如模板格式不匹配或API请求期间的问题。如果生成的模板无效，则返回到使用原始日志消息。
- 后期处理:后处理步骤（post_process_template）进一步细化生成的模板，确保占位符被适当替换，并且生成的模板对于匹配未来的日志是有效的。


###### 154 转 paring_cache.py 33 ParsingCache：事件模板的管理和匹配的类。它用于通过尝试将生成的模板与提供的log_message匹配来验证生成的模板。

###### 155 转 134 post_process_template
This function processes the generated template and adjusts it as necessary.

Process:
- Replaces {} placeholders in the template with <*>. 
- Applies the regular expressions in regs_common to the template. 
  - regs_common的用法：regs_common列表包含正则表达式，用于对生成的日志模板进行细化。可以对这些表达式进行定制，以处理特定日志格式的细微差别。
- Checks if the final template is valid and not too general.

Output: Returns the processed template if valid.

其中，139转post_process.py 9 correct_single_template,用于Apply all rules to process a template。

###### 157 转 paring_cache.py 46 add_templates
- 46 转 170 message_split
- 183 转 154 post_process_tokens
- 返回46
- 50 转 paring_cache.py 77 insert
- 52 return返回上级 gpt_query.py 157

###### 158 转 paring_cache.py 142 match_event
- 142 转 202 tree_match
- 204 转 170 message_split 
- 183 转 154 post_process_tokens
- 返回204
- 206 转 214 match_template
- 215 转 242 find_template （261）
- 返回返回返回到gpt_query.py 158

日志模板有效性：代码检查生成的模板是否有效，是否足够具体，可以用于进一步匹配。如果日志模板过于一般化，它将打印一条错误消息，并返回到原始日志消息。

Output: The processed log template if valid, or the original log message if no valid template is found.

##### 165 转 LILAC 151

##### 154 转 paring_cache.py 46 add_templates
同本文档298

##### 161 True 返回 117

##### 转 parsing_cache.py 141 match_event
同本文档200。

---------------------------------之后就卡这了-------------------------------------------------

#### 95-98
计算 FGA（过滤后的分组准确率），如果 PGA 或 RGA 非零，则使用 F1 比例公式计算 FGA。
返回GA,FGA

### 转evaluator_main.py 191

### 197-204
根据 lstm 参数的值，选择不同的解析准确率计算方式：
- 如果 lstm == True，调用 calculate_parsing_accuracy_lstm 计算 LSTM 模型的解析准确率（PA）。 (本例是if)
- 否则，调用传统的 calculate_parsing_accuracy 来计算普通的解析准确率。
- 记录计算解析准确率的时间并输出。

#### 202 转 PA_calculator.py, 64-74 calculate_parsing_accuracy.py
计算解析的准确度（PA）。

如果提供了 filter_templates，则只选取这些模板对应的数据进行比较。

使用 eq 方法比较 parsedresult_df 和 groundtruth_df 中的 EventTemplate 列，统计解析正确的消息数量。

将正确解析的消息数量除以总消息数，得到准确度（PA）。

最后返回准确度PA。

### 转203

### 206-213
计算模板级别的准确率。根据 lstm 参数选择使用 LSTM 或传统方法来计算模板级别的评估指标：
- 如果使用 LSTM (lstm == True)，调用 evaluate_template_level_lstm 来计算模板级别的评估。 （本例是if）
- 否则，调用 evaluate_template_level 来计算普通的模板级别评估。
- 记录计算模板级别准确率的时间，并打印输出。

#### 211 转 template_level_analysis.py 24 evaluate_template_level
该函数负责执行模板级别的分析，比较实际结果（df_groundtruth）和预测结果（df_parsedresult），并根据分类（SM, OG, UG, MX）进行评估。

函数的主要步骤：
- 过滤无效的日志：根据 EventTemplate 过滤掉空值（NaN）的日志。 
- 合并和分组数据：通过 EventTemplate 对实际结果和预测结果进行合并，并按预测结果分组。 
- 分析每个模板：遍历每个预测的模板（parsedlog），与实际结果（groundtruth）进行比对： 
- 如果预测的模板与实际的模板匹配，则视为正确解析。 
- 计算评估指标：根据正确解析的模板数量计算 PTA（模板准确率）、RTA（召回率）和 FTA（F1 分数）。

#### 37-67
- 过滤空值：df_groundtruth 和 df_parsedresult 中都进行过空值过滤，确保只有有效的模板参与后续分析。 
- 分组分析：通过 groupby 将 df_combined 按 parsedlog 分组，每一组代表一个预测的模板。 
- 匹配模板：如果预测的模板 identified_template 与实际的模板（corr_oracle_templates）完全一致，则认为该模板预测是正确的。

#### 71-84
- PTA（模板准确率）表示正确识别的模板占所有识别的模板的比例。 
- RTA（模板召回率）表示正确识别的模板占所有真实模板的比例。 
- FTA（F1 分数）是 PTA 和 RTA 的调和平均值。

### 转215

### 215-228
将当前评估结果整理为 CSV 格式的字符串，其中包含了：
- parse_time：解析所花费的时间。 
- tool_templates：解析工具识别出的模板数。 
- ground_templates：地面真值中的模板数。 
- GA：分组准确率（Grouping Accuracy）。 
- PA：解析准确率（Parsing Accuracy）。 
- FGA：过滤后的分组准确率（Filtered Grouping Accuracy）。 
- PTA：模板级别准确率（Partial Template Accuracy）。 
- RTA：根模板准确率（Root Template Accuracy）。 
- FTA：完全模板准确率（Full Template Accuracy）。

### 229-230
将构造好的结果字符串 result 写入到指定的输出文件 result_file 中，使用 a 模式表示追加到文件末尾。

## 返回92 evaluator

## 82开始轮换dataset，并重复上述过程（主要是92开始的evaluator）

## 109，110：后处理
- 在所有数据集的评估完成后，使用 post_average() 函数对评估结果进行后处理，计算平均值并保存。 
- metric_file 是评估结果文件的路径，post_average 使用此文件进行处理。 
- f"LILAC_{data_type}_complex={args.complex}_frequent={args.frequent}_{args.shot}_{args.example_size}_{args.model}" 是生成的后处理文件的名称，用于描述不同的参数设置。



