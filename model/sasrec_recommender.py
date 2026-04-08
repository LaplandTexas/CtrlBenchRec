import os
import torch
import numpy as np

from .config import STATE_DICT_KEY
from .sasrec import SASRec
from .narm import NARM
from .dataloader import dataloader_factory
from .utils import set_template, fix_random_seed_as, args


class SasRecRecommender:
    def __init__(self, custom_args=None):
        """
        初始化推荐器，加载模型。
        """
        self.args = custom_args if custom_args else args

        # 1. 环境初始化
        set_template(self.args)
        fix_random_seed_as(self.args.model_init_seed)
        self.device = torch.device(self.args.device)

        # 2. 构建模型并加载权重
        _, _, _ = dataloader_factory(self.args)

        self.args.model_code = "sas"
        self.model = self._build_model()
        self.model.to(self.device)
        self._load_weights()
        self.model.eval()

    def _build_model(self):
        if self.args.model_code == "sas":
            return SASRec(self.args)
        else:
            raise ValueError(f"Unknown model_code: {self.args.model_code}")

    def _load_weights(self):
        # 确保路径正确
        ckpt_path = "../model/sasrec/best_acc_model.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt[STATE_DICT_KEY])
        print(f"Successfully loaded model from: {ckpt_path}")

    @staticmethod
    def predict_custom_sequence(custom_ids, neg_ids=None, topk=20, model=None, model_args=None):
        """
        静态方法：针对 SASRec 优化的预测逻辑
        """
        target_model = model
        target_args = model_args if model_args else args

        target_model.eval()
        device = torch.device(target_args.device)

        # 1. 类型转换
        custom_ids_int = [int(x) for x in custom_ids]

        # 2. 预处理 (SASRec 推荐使用 args.bert_max_len 或 args.max_len)
        # 注意：这里建议使用配置中的长度，而不是硬编码 150
        max_len = getattr(target_args, 'bert_max_len', 150)
        if len(custom_ids_int) > max_len:
            seq = custom_ids_int[-max_len:]
        else:
            seq = [0] * (max_len - len(custom_ids_int)) + custom_ids_int

        # 3. 构造 Tensor (SASRec 只需要 seq_tensor)
        seq_tensor = torch.LongTensor([seq]).to(device)

        with torch.no_grad():
            # 4. 前向计算
            all_scores = target_model(seq_tensor)

            # --- 关键修改：如果维度是 [batch, seq_len, items]，只取最后一个时间步 ---
            if len(all_scores.shape) == 3:
                all_scores = all_scores[:, -1, :]  # 形状变为 [1, num_items]
            # -----------------------------------------------------------

            # 5. 执行负过滤
            if neg_ids is not None:
                neg_ids_int = [int(x) for x in neg_ids]
                neg_ids_int = [x for x in neg_ids_int if x < all_scores.size(1)]
                all_scores[0, neg_ids_int] = -1e9

            # 6. 获取 Top-K
            k = min(topk, all_scores.size(1))
            scores, indices = all_scores.topk(k=k, dim=-1)

        # 此时 indices[0] 只有 1 组 top-k 结果
        return indices[0].tolist(), scores[0].tolist()

    def recommend(self, custom_ids, neg_ids=None, topk=3):
        return self.predict_custom_sequence(
            custom_ids=custom_ids,
            neg_ids=neg_ids,
            topk=topk,
            model=self.model,
            model_args=self.args
        )