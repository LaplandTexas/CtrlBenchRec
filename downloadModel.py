# 使用
from modelscope import AutoModel, AutoTokenizer

if __name__ == '__main__':
    model = AutoModel.from_pretrained(
        "qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "qwen/Qwen2.5-7B-Instruct",
        trust_remote_code=True
    )