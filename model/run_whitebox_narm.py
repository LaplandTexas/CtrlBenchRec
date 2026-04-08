import os
import torch

from .config import STATE_DICT_KEY
from .sasrec import SASRec
from .narm import NARM
from .dataloader import dataloader_factory
from datasets import DATASETS
from .utils import set_template, fix_random_seed_as, args


def build_model(args):
    """根据 args.model_code 创建模型实例。"""
    if args.model_code == "sas":
        model = SASRec(args)
    elif args.model_code == "narm":
        model = NARM(args)
    else:
        raise ValueError(f"Unknown model_code: {args.model_code}")
    return model

def load_whitebox_model(args):
    """
    加载蒸馏后的白盒模型（best_acc_model.pth）。

    folder_name 例如：'narm2narm_autoregressive5000'
    """
    device = torch.device(args.device)

    # 先构建 dataloader，这一步会通过 dataset_factory 设置 args.num_items 等信息
    train_loader, val_loader, test_loader = dataloader_factory(args)

    # 构建模型
    model = build_model(args)
    model.to(device)

    # 蒸馏模型保存的路径：experiments/distillation_rank/<folder_name>/<dataset_code>/models/best_acc_model.pth
    ckpt_path = "narm/best_acc_model.pth"

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt[STATE_DICT_KEY])
    model.eval()

    print(f"Loaded white-box model from: {ckpt_path}")
    return model, test_loader


def run_one_batch(model, test_loader, args, topk=10):
    """
    从 test_loader 里取一个 batch，跑一遍模型并打印结果。
    """
    device = torch.device(args.device)

    # 取第一个 batch
    batch = next(iter(test_loader))
    # 原项目的 dataloader 在 Profile Pollution 代码解读中是这样的结构：
    #   seqs, candidates, labels = batch
    seqs, candidates, labels, *unused = batch
    seqs = seqs.to(device)           # [B, L]
    candidates = candidates.to(device)  # [B, num_candidates]
    labels = labels.to(device)       # [B]

    # 计算每个用户序列的真实长度（非 0 的位置）
    lengths = (seqs > 0).sum(dim=1)  # [B]

    with torch.no_grad():
        # NARM.forward(x, lengths) 返回对所有 item 的打分 [B, num_items]
        all_scores = model(seqs, lengths)

        # 只取候选集上的得分
        batch_size, num_cand = candidates.shape
        batch_idx = torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, num_cand)
        cand_scores = all_scores[batch_idx, candidates]  # [B, num_cand]

        # 在候选集上取 topk
        k = min(topk, num_cand)
        topk_scores, topk_indices = cand_scores.topk(k=k, dim=-1)
        topk_items = candidates[batch_idx, topk_indices]  # [B, k]

    # 打印第一个用户的结果看一下
    u = 0
    user_seq = seqs[u].detach().cpu().tolist()
    user_seq = [x for x in user_seq if x > 0]  # 去掉 padding 0

    print("\n========== Example Output ==========")
    print(f"Dataset      : {args.dataset_code}")
    print(f"Model        : {args.model_code} (white-box, distilled)")
    print(f"User 0 seq   : {user_seq}")
    print(f"Ground-truth : {int(labels[u].item())}")
    print(f"Top-{k} cand : {topk_items[u].detach().cpu().tolist()}")
    print("====================================\n")


def predict_custom_sequence(model, custom_ids, args, topk=3, neg_ids=None):
    """
    custom_ids: 你的自定义序列，例如 [101, 50, 200]
    neg_ids: 负例列表（不希望出现的物品 ID），例如 ["5", "99"] 或 [5, 99]
    """
    model.eval()
    device = torch.device(args.device)

    # 转换 custom_ids 为整数列表
    custom_ids_raw = custom_ids
    custom_ids = []
    for custom_id_str in custom_ids_raw:
        custom_ids.append(int(custom_id_str))

    # --- 新增：参考原本的逻辑，将 neg_ids 也转化为整数列表 ---
    neg_ids_final = []
    if neg_ids is not None:
        for neg_id_raw in neg_ids:
            neg_ids_final.append(int(neg_id_raw))
    # --------------------------------------------------

    # 1. 预处理：满足模型要求的 max_len
    max_len = 100
    if len(custom_ids) > max_len:
        seq = custom_ids[-max_len:]  # 截断
    else:
        seq = [0] * (max_len - len(custom_ids)) + custom_ids  # 左侧填充 0

    # 2. 转换为 Tensor
    seq_tensor = torch.LongTensor([seq]).to(device)  # 形状 [1, max_len]
    length_tensor = torch.LongTensor([len(custom_ids)]).to(device)  # 形状 [1]

    with torch.no_grad():
        # 3. 前向计算得到所有物品的分数
        all_scores = model(seq_tensor, length_tensor)  # [1, num_items]

        # --- 执行负过滤 ---
        if neg_ids_final:
            # 使用 -1e9 将这些负例物品的分数压到极低
            all_scores[0, neg_ids_final] = -1e9
        # -----------------

        # 4. 获取 Top-K
        scores, indices = all_scores.topk(k=topk, dim=-1)

    return indices[0].tolist(), scores[0].tolist()


def main():
    # 1. 用原来的交互方式设置超参 / 数据集 / 模型
    print(">>> Setting template (same as train.py / distill.py)")
    set_template(args)

    # 这里建议你选择：
    #   - 输入 '1' → dataset_code = 'ml-1m'
    #   - 输入 'n' → model_code = 'narm'
    #   - GPU ID 和之前训练/蒸馏时保持一致（或者直接用 cpu）

    fix_random_seed_as(args.model_init_seed)

    # 3. 加载白盒替代模型 + dataloader
    model, test_loader = load_whitebox_model(args)

    custom_ids = ['1046','2213','45','668']

    print(predict_custom_sequence(model, custom_ids, args))

if __name__ == "__main__":
    main()
