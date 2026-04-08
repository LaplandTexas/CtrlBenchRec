from transformers import AutoTokenizer, AutoModelForMaskedLM
import os
from tool.path_constants import PathConstants

tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path= PathConstants.twhin_bert_base_model_path)
model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path= PathConstants.twhin_bert_base_model_path)

# 3. 如果文件夹不存在，先创建它
if not os.path.exists(PathConstants.twhin_bert_base_model_path):
    os.makedirs(PathConstants.twhin_bert_base_model_path)

# 4. 执行存放（保存）命令
tokenizer.save_pretrained(PathConstants.twhin_bert_base_model_path)
model.save_pretrained(PathConstants.twhin_bert_base_model_path)

print(f"load twhin-bert-sucessfully!Path：{os.path.abspath(PathConstants.twhin_bert_base_model_path)}")