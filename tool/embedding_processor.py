import json

import numpy as np
from torch.nn.functional import cosine_similarity

from tool.movie_len_loader import MovieLenLoader
from tool.path_constants import PathConstants


class EmbeddingProcessor:

    @staticmethod
    def save_embedding(user_vector, post_vector, embedding_file_path ='..\\embeddings\\embeddings.npz'):
        # 假设的用户profile embedding (用户特征向量)
        user_profile_embedding = user_vector  # 128维用户向量

        # 假设的item list embedding (物品特征向量列表)
        item_list_embedding = post_vector  # 50个物品，每个64维

        print(f"user_profile_embedding.shape:{user_profile_embedding.shape}")
        print(f"item_list_embedding.shape:{item_list_embedding.shape}")

        # 方法1：使用np.savez保存多个数组到单个文件
        np.savez(embedding_file_path,
                 user_profile=user_profile_embedding,
                 item_list=item_list_embedding)

        print("Embedding数据已保存到 embeddings.npz")

    @staticmethod
    def load_embedding(embedding_file_path):
        ## 读取embedding数据
        # 从npz文件中读取数据
        data = np.load(embedding_file_path)

        # 获取用户profile embedding
        user_profile = data['user_profile']
        # 获取item list embedding
        item_list = data['item_list']

        print("成功读取embedding数据:")
        print(f"用户profile形状: {user_profile.shape}")
        print(f"物品列表形状: {item_list.shape}")
        print(f"用户profile前5个值: {user_profile[:5]}")
        return (user_profile, item_list)

    @staticmethod
    def load_behavior_embedding(behavior_file_path):
        data = np.load(behavior_file_path)

        user_behavior_vector = data['user_behavior']
        return user_behavior_vector

    @staticmethod
    def save_behavior_embedding(user_behavior_vector_all,embedding_file_path ='..\\embeddings\\behavior_embeddings.npz'):
        print(f"user_behavior_vector_all.shape:{user_behavior_vector_all.shape}")
        np.savez(embedding_file_path,user_behavior=user_behavior_vector_all)
        print("用户行为流Embedding已保存到..\\embeddings\\behavior_embeddings.npz")

    @staticmethod
    def get_embeddings_by_target_tag(target_tag, raw_embedding_path, storage_path):
        # 1. 加载数据
        (user_profile, item_list) = EmbeddingProcessor.load_embedding(raw_embedding_path)
        # 强制转为 numpy array 方便后续矩阵运算
        all_embeddings = np.array(item_list)

        # 2. 计算全局平均值 (解决相似度过高的关键)
        # 减去这个“背景向量”能让不同标签的特征凸显出来
        global_mean = np.mean(all_embeddings, axis=0)
        with open("../embeddings/task2/all_post_average/global.json", 'w', encoding='utf-8') as f:
            json.dump(global_mean.tolist(), f, indent=4)

        movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
        target_item_list = []

        # 3. 精确匹配标签
        for movie_id, movie_info in movie_id_and_info_map.items():
            genres = movie_info.get('genres', '').split('|')

            if target_tag in genres:
                try:
                    # 确保 movie_id 对应正确的 item_list 索引
                    # 注意：如果 ID 不是从 0 开始的连续整数，这里需要根据你的 ID 映射逻辑修改
                    idx = int(movie_id)-1
                    if idx < len(all_embeddings):
                        # 减去全局均值，进行中心化处理
                        target_item_list.append(all_embeddings[idx] - global_mean)
                except (ValueError, IndexError):
                    continue

        # 4. 结果处理
        if not target_item_list:
            print(f"Warning: No embeddings found for tag '{target_tag}'")
            return []

        target_item_list_embeddings = np.array(target_item_list)

        # 求平均，并保持维度 (1, 768)
        result_nodes = np.mean(target_item_list_embeddings, axis=0, keepdims=True)

        # 转换为 list 供 JSON 序列化
        result = result_nodes.tolist()

        if storage_path is not None:
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)

        return result

    @staticmethod
    def get_embeddings_by_user_profile(user_profile_path,raw_embedding_path, storage_path):
        # 1. 加载数据
        (user_profile, item_list) = EmbeddingProcessor.load_embedding(raw_embedding_path)
        # 强制转为 numpy array 方便后续矩阵运算
        all_embeddings = np.array(item_list)

        # 2. 计算全局平均值 (解决相似度过高的关键)
        # 减去这个“背景向量”能让不同标签的特征凸显出来
        global_mean = np.mean(all_embeddings, axis=0)

        with open(user_profile_path, 'r', encoding='utf-8') as f:
            user_profile_list = json.load(f)

        user_profile = user_profile_list[1]
        behavior_list = user_profile['behavior']

        target_item_list = []
        for behavior in behavior_list:
            try:
                # 确保 movie_id 对应正确的 item_list 索引
                # 注意：如果 ID 不是从 0 开始的连续整数，这里需要根据你的 ID 映射逻辑修改
                idx = int(behavior) - 1
                if idx < len(all_embeddings):
                    # 减去全局均值，进行中心化处理
                    target_item_list.append(all_embeddings[idx] - global_mean)
            except (ValueError, IndexError):
                continue

        # 4. 结果处理
        target_item_list_embeddings = np.array(target_item_list)

        # 求平均，并保持维度 (1, 768)
        result_nodes = np.mean(target_item_list_embeddings, axis=0, keepdims=True)

        # 转换为 list 供 JSON 序列化
        result = result_nodes.tolist()

        if storage_path is not None:
            with open(storage_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=4)

        return result

    @staticmethod
    def get_all_embeddings_by_user_profile(user_profile_path, raw_embedding_path, storage_path):
        # 1. 加载数据
        (_, item_list) = EmbeddingProcessor.load_embedding(raw_embedding_path)
        all_embeddings = np.array(item_list)

        # 2. 计算全局平均值 (中心化处理)
        global_mean = np.mean(all_embeddings, axis=0)

        # 3. 加载用户画像列表
        with open(user_profile_path, 'r', encoding='utf-8') as f:
            user_profile_list = json.load(f)

        # 用于存储所有用户的 embedding 结果
        # 格式建议为 {user_id: embedding_list}
        all_user_embeddings_map = {}

        # 4. 遍历所有用户
        for user_profile in user_profile_list:
            user_id = str(user_profile.get('user_id', len(all_user_embeddings_map)))
            behavior_list = user_profile.get('behavior', [])

            if not behavior_list:
                # 如果该用户没有行为记录，可以跳过或填充全 0 向量
                continue

            target_item_list = []
            for behavior in behavior_list:
                try:
                    # 假设 ID 映射逻辑：int(id) - 1
                    idx = int(behavior) - 1
                    if 0 <= idx < len(all_embeddings):
                        # 中心化处理：减去背景向量
                        target_item_list.append(all_embeddings[idx] - global_mean)
                except (ValueError, IndexError):
                    continue

            # 5. 计算该用户的平均 embedding
            if target_item_list:
                target_item_list_embeddings = np.array(target_item_list)
                # 计算均值 (shape: (768,))
                user_mean_embedding = np.mean(target_item_list_embeddings, axis=0)
                # 存入字典，转为 list 方便 JSON 序列化
                all_user_embeddings_map[user_id] = user_mean_embedding.tolist()

        # 6. 存储结果
        if storage_path is not None:
            with open(storage_path, 'w', encoding='utf-8') as f:
                # 使用 json.dump 存储整个字典
                json.dump(all_user_embeddings_map, f, indent=4)

        return all_user_embeddings_map

if __name__ == "__main__":
    # EmbeddingProcessor.load_behavior_embedding("../embeddings/task2/l1_user_embedding/0.npz")
    # EmbeddingProcessor.get_embeddings_by_target_tag("Action",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/action.json")
    # EmbeddingProcessor.get_embeddings_by_target_tag("Adventure",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/adventure.json")
    # EmbeddingProcessor.get_embeddings_by_target_tag("Thriller",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/thriller.json")
    # EmbeddingProcessor.get_embeddings_by_target_tag("Romance",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/romance.json")
    # EmbeddingProcessor.get_embeddings_by_target_tag("War",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/war.json")
    # EmbeddingProcessor.get_embeddings_by_target_tag("Film-Noir",
    #                                                 "../embeddings/embeddings.npz",
    #                                                 "../embeddings/task2/all_post_average/film-noir.json")
    # EmbeddingProcessor.get_embeddings_by_user_profile("../generated_user_profile/task2/l1_change_profile_and_prompt/user_1_profile_0_19.json",
    #                                                   "../embeddings/task2/all_post_average/embeddings.npz",
    #                                                   "../embeddings/task2/l1_user_embedding/20.json")
    EmbeddingProcessor.get_all_embeddings_by_user_profile("../generated_user_profile/task2/l3/user_28_profile_0_17.json",
                                                          "../embeddings/task2/all_post_average/embeddings.npz",
                                                          "../embeddings/task2/l3_user_embedding/18.json")