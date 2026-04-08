import os
import torch
import numpy as np

from .config import STATE_DICT_KEY
from .sasrec import SASRec
from .narm import NARM
from .dataloader import dataloader_factory
from .utils import set_template, fix_random_seed_as, args


class NarmRecommender:
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
        # 初始化 dataloader 以填充 args 中的 num_items 等元数据
        _, _, _ = dataloader_factory(self.args)

        self.model = self._build_model()
        self.model.to(self.device)
        self._load_weights()
        self.model.eval()

    def _build_model(self):
        if self.args.model_code == "sas":
            return SASRec(self.args)
        elif self.args.model_code == "narm":
            return NARM(self.args)
        else:
            raise ValueError(f"Unknown model_code: {self.args.model_code}")

    def _load_weights(self):
        ckpt_path = "../model/narm/best_acc_model.pth"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        ckpt = torch.load(ckpt_path, map_location=self.device)
        self.model.load_state_dict(ckpt[STATE_DICT_KEY])
        print(f"Successfully loaded model from: {ckpt_path}")

    @staticmethod
    def predict_custom_sequence(custom_ids, neg_ids=None, topk=20, model=None, model_args=None):
        """
        静态方法：根据给定序列预测 Top-K 物品。
        :param custom_ids: 原始 ID 列表
        :param neg_ids: 黑名单 ID 列表
        :param topk: 推荐数量
        :param model: 模型实例 (可选，默认使用 None)
        :param model_args: 配置参数 (可选，默认使用 None)
        """
        # 如果调用时没传 model 或 args，尝试使用全局的或外部传入的
        # 注意：在静态方法内部直接访问 self 是不可能的，所以需要外部传入或使用默认配置
        target_model = model
        target_args = model_args if model_args else args

        target_model.eval()
        device = torch.device(target_args.device)

        # 1. 类型转换
        custom_ids_int = [int(x) for x in custom_ids]

        # 2. 预处理 (左侧填充 0)
        max_len = 150
        if len(custom_ids_int) > max_len:
            seq = custom_ids_int[-max_len:]
        else:
            seq = [0] * (max_len - len(custom_ids_int)) + custom_ids_int

        # 3. 构造 Tensor
        seq_tensor = torch.LongTensor([seq]).to(device)
        length_tensor = torch.LongTensor([len(custom_ids_int)]).to(device)

        with torch.no_grad():
            # 4. 前向计算
            all_scores = target_model(seq_tensor, length_tensor)

            # 5. 执行负过滤
            if neg_ids is not None:
                neg_ids_int = [int(x) for x in neg_ids]
                neg_ids_int = [x for x in neg_ids_int if x < all_scores.size(1)]
                all_scores[0, neg_ids_int] = -1e9

            # 6. 获取 Top-K
            k = min(topk, all_scores.size(1))
            scores, indices = all_scores.topk(k=k, dim=-1)

        return indices[0].tolist(), scores[0].tolist()

    def recommend(self, custom_ids, neg_ids=None, topk=3):
        """
        实例方法：直接封装静态方法，并传入自身的 model 和 args
        """
        return self.predict_custom_sequence(
            custom_ids=custom_ids,
            neg_ids=neg_ids,
            topk=topk,
            model=self.model,
            model_args=self.args
        )


# --- 测试代码 ---
if __name__ == "__main__":
    recommender = NarmRecommender()

    test_ids = ['1046', '2213', '45', '668']
    blacklist = ['2720']

    # 现在的调用非常简洁：只需提供原始ID和黑名单ID
    items, scores = recommender.recommend(test_ids, neg_ids=blacklist, topk=3)

    print("\n>>>> Custom Recommendation Results <<<<")
    print(f"Items  : {items}")
    print(f"Scores : {scores}")