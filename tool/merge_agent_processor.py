import json
import sqlite3
from collections import Counter

import numpy as np
from openai import OpenAI
from sklearn.mixture import GaussianMixture

from tool.amazon_loader import AmazonLoader
from tool.embedding_processor import EmbeddingProcessor

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from tool.movie_len_loader import MovieLenLoader
from tool.token_statistic_processor import TokenStatisticProcessor


class MergeAgentProcessor:

    @staticmethod
    def calculate_iou_similarity(user_matrix):
        """
        计算用户间的IOU相似度
        user_matrix: 100×4000的用户-内容交互矩阵（二值化）
        """
        n_users = user_matrix.shape[0]
        iou_matrix = np.zeros((n_users, n_users))

        # 将交互矩阵二值化（如果还不是二值）
        binary_matrix = (user_matrix > 0).astype(int)

        for i in range(n_users):
            for j in range(i + 1, n_users):
                # 用户i和j的交互集合
                set_i = set(np.where(binary_matrix[i] > 0)[0])
                set_j = set(np.where(binary_matrix[j] > 0)[0])

                # 计算IOU
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)

                iou = intersection / union if union > 0 else 0
                iou_matrix[i][j] = iou
                iou_matrix[j][i] = iou
        return iou_matrix

    @staticmethod
    def cosine_similarity_matrix(embeddings):
        """计算embedding矩阵的余弦相似度矩阵"""
        # 归一化
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_normalized = embeddings / norms
        # 计算相似度矩阵
        similarity_matrix = np.dot(embeddings_normalized, embeddings_normalized.T)
        # similarity_matrix = np.dot(embeddings, embeddings.T)
        return similarity_matrix

    @staticmethod
    def get_user_id_and_like_map(db_path = '..\\data\\twitter_simulation.db'):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM like")
        results = cursor.fetchall()
        user_id_and_like_map = {}
        for result in results:
            user_id = result[1]
            like_list = []
            if user_id in user_id_and_like_map:
                like_list = user_id_and_like_map[user_id]
            like_list.append(str(result[2]))
            user_id_and_like_map[user_id] = like_list
        return user_id_and_like_map

    @staticmethod
    def get_user_id_and_rec_map(db_path = '../data/twitter_simulation.db'):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM rec")
        results = cursor.fetchall()
        user_id_and_rec_map = {}
        for result in results:
            user_id = result[0]
            rec_list = []
            if user_id in user_id_and_rec_map:
                rec_list = user_id_and_rec_map[user_id]
            rec_list.append(str(result[1]))
            user_id_and_rec_map[user_id] = rec_list
        return user_id_and_rec_map

    @staticmethod
    def get_group_id_and_chat_history_map(db_path = '../data/twitter_simulation.db'):
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM group_messages order by group_id asc")
        results = cursor.fetchall()
        group_id_and_chat_history_map = {}
        for result in results:
            group_id = result[1]
            user_id = result[2]+1
            content = result[3]
            chat_history = {}
            if group_id in group_id_and_chat_history_map:
                chat_history_list = group_id_and_chat_history_map[group_id]
            else:
                chat_history_list = []
            chat_history['user_id'] = user_id
            chat_history['content'] = content
            chat_history_list.append(chat_history)
            group_id_and_chat_history_map[group_id] = chat_history_list
        return group_id_and_chat_history_map

    @staticmethod
    def get_user_behavior_vector(user_id_and_like_map,user_vector,posts_vector):
        user_behavior_vector_all = []
        for user_idx in user_id_and_like_map.keys():
            like_post_ids = user_id_and_like_map[user_idx]
            like_posts_vector = []
            for like_post_id in like_post_ids:
                try:
                    like_posts_vector.append(posts_vector[like_post_id - 1])
                    # rec_log.info(f"like_post_id:{like_post_id}")
                except Exception:
                    like_posts_vector.append(user_vector[user_idx])
            mean_like_posts_vector = np.mean(np.array(like_posts_vector), axis=0)
            user_behavior_vector = (0.2* user_vector[user_idx] + 0.8 * mean_like_posts_vector)
            user_behavior_vector_all.append(user_behavior_vector)
        return user_behavior_vector_all

    @staticmethod
    def kmeans_get_merged_new_agent_id_and_origin_id_map(user_embeddings, max_clusters=None):
        """
        自动确定最佳聚类数的简化实现
        """
        if max_clusters is None:
            max_clusters = max(round(len(user_embeddings)*1.0/2), round(len(user_embeddings)*1.0/4)*3)

        best_score = -1
        best_k = 0
        best_labels = None
        best_model = None

        # 尝试不同的k值
        for k in range(round(len(user_embeddings)*1.0/2), max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
            labels = kmeans.fit_predict(user_embeddings)

            # 计算轮廓系数
            if len(np.unique(labels)) > 1:
                score = silhouette_score(user_embeddings, labels)

                if score > best_score:
                    best_score = score
                    best_k = k
                    best_labels = labels
                    best_model = kmeans
                print(f"score: {score}, k: {k}, labels: {labels},unique_labels_len:{len(np.unique(labels))}")
                label_counts = Counter(labels)  # 自动统计每个标签的出现次数
                print("标签计数结果:", dict(label_counts))

        print(f"最佳聚类数: {best_k}, 轮廓系数: {best_score:.4f}")

        new_agent_id_and_origin_id_map = {}
        for index in range(len(best_labels)):
            label = int(best_labels[index])
            origin_id_list = []
            if label in new_agent_id_and_origin_id_map:
                origin_id_list = new_agent_id_and_origin_id_map[label]
            origin_id_list.append(index)
            new_agent_id_and_origin_id_map[label] = origin_id_list

        return new_agent_id_and_origin_id_map

    @staticmethod
    def gmm_get_agent_cluster_probabilities(user_embeddings, max_clusters=None):
        """
        使用高斯混合模型 (GMM) 自动确定最佳聚类数（基于 BIC），
        并返回智能体对每个簇的概率隶属度矩阵 (Soft Assignment)。

        Args:
            user_embeddings (np.array): 智能体的 embedding 矩阵。
            max_clusters (int, optional): 搜索的最大簇数量。

        Returns:
            np.array: 概率隶属度矩阵 (N_agents x K_best)。
            int: 最佳聚类数 K_best。
        """
        N = len(user_embeddings)
        if N < 2:
            print("警告：智能体数量少于2，无法进行聚类。")
            return np.ones((N, 1)), 1

        # 定义搜索范围（与原代码逻辑相似，但确保 min_k >= 2）
        min_k = max(2, round(N * 1.0 / 2))
        if max_clusters is None:
            max_k = round(N * 3.0 / 4)  # 默认使用 N3/4 作为上限
        else:
            max_k = max_clusters

        best_score = np.inf  # GMM 通常使用 BIC/AIC，目标是最小化 (越小越好)
        best_k = min_k
        best_model = None

        # 尝试不同的k值
        for k in range(min_k, min_k + 1):
            try:
                user_embeddings = np.array(user_embeddings)
                # GMM 模型初始化，使用 BIC/AIC 推荐的 'full' 协方差类型
                gmm = GaussianMixture(
                    n_components=k,
                    random_state=42,
                    n_init=5,
                    covariance_type='full'  # 修正 GMM 退化问题
                )
                gmm.fit(user_embeddings)

                # 使用 BIC (Bayesian Information Criterion) 作为模型选择指标
                score = gmm.bic(np.array(user_embeddings))

                # 由于 BIC 越小模型越好，我们寻找最小的 BIC
                if score < best_score:
                    best_score = score
                    best_k = k
                    best_model = gmm

                print(f"BIC score: {score:.4f}, k: {k}")

            except ValueError as e:
                # 捕获 GMM 可能不收敛的错误
                print(f"k={k} 时 GMM 拟合失败: {e}")
                continue

        print(f"--- GMM 聚类结果 ---")
        print(f"最佳聚类数: {best_k}, 最佳 BIC score: {best_score:.4f}")

        if best_model is None:
            # 应对所有拟合都失败的极端情况
            raise RuntimeError("GMM 拟合失败，无法确定最佳聚类数。")

        # --- 获取概率隶属度 ---
        # predict_proba() 返回 N_agents x K_best 矩阵
        # 这是软聚类的核心输出，用作 LLM 融合时的权重。
        probabilities = best_model.predict_proba(user_embeddings)

        # 注意：不再返回 new_agent_id_and_origin_id_map，因为这是硬分配的概念。
        # 融合逻辑将在外部使用这个 'probabilities' 矩阵。

        return probabilities, best_k

    @staticmethod
    def load_user_profile_with_extra_info(user_profile_path = "..\\generated_user_profile\\user_100_profile.json"):
        movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
        agent_info_list = []
        with open(user_profile_path, "r", encoding='utf-8') as file:
            agent_info_list = json.load(file)
        for index in range(len(agent_info_list)):
            agent_info = agent_info_list[index]
            behavior_list = agent_info['behavior']
            for j in range(len(behavior_list)):
                movie_id = str(behavior_list[j])
                if movie_id not in movie_id_and_info_map:
                    continue
                movie_info = movie_id_and_info_map[movie_id]
                movie_name = movie_info['title']
                movie_genres = movie_info['genres']
                behavior_list[j] = str(behavior_list[j])+f":{movie_name},({movie_genres})"
            agent_info['behavior'] = behavior_list
            agent_info_list[index] = agent_info

        return agent_info_list

    @staticmethod
    def load_user_profile_with_extra_info_amazon(user_profile_path):
        user_id_and_item_id_map = AmazonLoader.get_user_id_and_item_id_map()
        item_id_and_info_map = AmazonLoader.get_item_id_and_info_map()
        agent_info_list = []
        with open(user_profile_path, "r", encoding='utf-8') as file:
            agent_info_list = json.load(file)
        for index in range(len(agent_info_list)):
            agent_info = agent_info_list[index]
            behavior_list = agent_info['behavior']
            for j in range(len(behavior_list)):
                item_id = str(behavior_list[j])
                item_info =item_id_and_info_map[item_id]
                item_title = item_info['Title']
                item_description = item_info['Description']
                item_category = item_info['Category']
                behavior_list[j] = str(behavior_list[j])+f":{{\"Title\":\"{item_title}\",\n\"Description\":\"{item_description}\",\n\"Category\":\"{item_category}\"}}"
            agent_info['behavior'] = behavior_list
            agent_info_list[index] = agent_info
        print(agent_info_list)
        return agent_info_list

    @staticmethod
    def get_merged_new_user(new_user_id, origin_agent_info_list_str,target_tag_list = []):
        origin_agent_info_list = json.loads(origin_agent_info_list_str)
        is_single = False
        if len(origin_agent_info_list) == 1:
            is_single = True

        client = OpenAI(
            base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
            api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
        )

        system_prompt = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with several user profiles. Please strictly follow the requirements below, comprehensively consider these user profiles, and synthesize them into a new user profile.",

            "# INPUT EXAMPLE #": {
                "userName": "",  # Username
                "bio": "",  # User biography
                "persona": "",  # User personality
                "age": "",  # User age
                "gender": "",  # User gender,only can be "Male" or "Female"
                "mbti": "",  # User MBTI personality type
                "profession": "",  # User profession
                "agent_behavioral_type": "",
                # User content exploration behavior type, can be random, focused, or exploratory
                "interested_topics": [],  # Movie topics user is interested in
                "behavior": []  # User behavior history, format: 'movie_id:movie_name,(genre_tags)'
            },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"Combine multiple users' \"bio\" to create a new \"bio\", aiming to maximally cover the original bio characteristics of multiple users.While trying to retain interests related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization."
                },
                {
                    "step": 3,
                    "description": f"Extract the top 5 most frequent \"interested_topics\" from multiple users' interested_topics to form new interested_topics"
                },
                {
                    "step": 4,
                    "description": f"For each user's \"behavior\", filter out movies that are relatively irrelevant to the new user's characteristics.While trying to retain movies related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only movie IDs and package them into a new JSON list as the new user's \"behavior\""
                },
                {
                    "step": 5,
                    "description": "Fill in other attributes of the user profile based on the newly synthesized \"bio\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id'
                }
            }
        }

        system_prompt_single = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with a user profile. Please strictly follow the requirements below, comprehensively synthesize it into a new user profile.",

            "# INPUT EXAMPLE #": {
                "userName": "",  # Username
                "bio": "",  # User biography
                "persona": "",  # User personality
                "age": "",  # User age
                "gender": "",  # User gender,only can be "Male" or "Female"
                "mbti": "",  # User MBTI personality type
                "profession": "",  # User profession
                "agent_behavioral_type": "",
                # User content exploration behavior type, can be random, focused, or exploratory
                "interested_topics": [],  # Movie topics user is interested in
                "behavior": []  # User behavior history, format: 'movie_id:movie_name,(genre_tags)'
            },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"For each user's \"behavior\", filter out movies that are relatively irrelevant to the new user's characteristics.While trying to retain movies related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only movie IDs and package them into a new JSON list as the new user's \"behavior\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id'
                }
            }
        }

        response = ''
        if is_single:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"{system_prompt_single}"},
                    {"role": "user", "content": origin_agent_info_list_str}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
        else:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": origin_agent_info_list_str}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )

        return json.loads(response.choices[0].message.content)

    @staticmethod
    def get_merged_new_user_with_chat_history(new_user_id, origin_agent_info_list_str, chat_history_path,target_tag_list=[]):
        origin_agent_info_list = json.loads(origin_agent_info_list_str)
        with open(chat_history_path, "r", encoding="utf-8") as f:
            group_id_and_chat_history_map = json.load(f)
        chat_history = group_id_and_chat_history_map.get(str(new_user_id))
        input = {}
        input['user_profile_list']=origin_agent_info_list
        input['chat_history_list']=chat_history
        input = json.dumps(input)
        is_single = False
        if len(origin_agent_info_list) == 1:
            is_single = True

        client = OpenAI(
            base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
            api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
        )

        #TODO 修改system_prompt和正常prompt，使其可以通过用户profile+聊天记录合成新用户
        system_prompt = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with several user profiles and their chat history. Please strictly follow the requirements below, comprehensively consider these user profiles, and synthesize them into a new user profile.",

            "# INPUT EXAMPLE #":
                {"user_profile_list":[
                    {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id:movie_name,(genre_tags)'
                    }],
                "chat_history_list":[
                    {
                        "user_id": "",# as same as userName
                        "content":""# User's chat history.After the @ symbol, the corresponding ID is userName-1. For example, @1 actually @'s the user with userName 2.
                    }
                ]
                },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"Based on users' chat history,combine multiple users' \"bio\" to create a new \"bio\", aiming to maximally cover the original bio characteristics of multiple users.While trying to retain interests related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization."
                },
                {
                    "step": 3,
                    "description": f"Extract the top 5 most frequent \"interested_topics\" from multiple users' interested_topics to form new interested_topics"
                },
                {
                    "step": 4,
                    "description": f"For each user's \"behavior\", filter out movies that are relatively irrelevant to the new user's characteristics and their chat history.While trying to retain movies related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only movie IDs and package them into a new JSON list as the new user's \"behavior\""
                },
                {
                    "step": 5,
                    "description": "Fill in other attributes of the user profile based on the newly synthesized \"bio\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id'
                }
            }
        }

        system_prompt_single = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with a user profile. Please strictly follow the requirements below, comprehensively synthesize it into a new user profile.",

            "# INPUT EXAMPLE #":
                {"user_profile_list":[
                    {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id:movie_name,(genre_tags)'
                    }],
                "chat_history_list":[
                    {
                        "user_id": "",# as same as userName
                        "content":""# User's chat history.After the @ symbol, the corresponding ID is userName-1. For example, @1 actually @'s the user with userName 2.
                    }
                ]
                },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"For each user's \"behavior\", filter out movies that are relatively irrelevant to the new user's characteristics.While trying to retain movies related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only movie IDs and package them into a new JSON list as the new user's \"behavior\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "persona": "",  # User personality
                    "age": "",  # User age
                    "gender": "",  # User gender,only can be "Male" or "Female"
                    "mbti": "",  # User MBTI personality type
                    "profession": "",  # User profession
                    "agent_behavioral_type": "",
                    # User content exploration behavior type, can be random, focused, or exploratory
                    "interested_topics": [],  # Movie topics user is interested in
                    "behavior": []  # User behavior history, format: 'movie_id'
                }
            }
        }

        response = ''
        if is_single:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"{system_prompt_single}"},
                    {"role": "user", "content": input}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )
        else:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": f"{system_prompt}"},
                    {"role": "user", "content": input}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}  # 强制 JSON 输出
            )

        return json.loads(response.choices[0].message.content)

    @staticmethod
    def get_merged_new_user_with_chat_history_amazon(new_user_id, origin_agent_info_list_str, chat_history_path,target_tag_list=[]):
        origin_agent_info_list = json.loads(origin_agent_info_list_str)
        with open(chat_history_path, "r", encoding="utf-8") as f:
            group_id_and_chat_history_map = json.load(f)
        chat_history = group_id_and_chat_history_map.get(str(new_user_id))
        input = {}
        input['user_profile_list']=origin_agent_info_list
        input['chat_history_list']=chat_history
        input = json.dumps(input)
        is_single = False
        if len(origin_agent_info_list) == 1:
            is_single = True

        client = OpenAI(
            base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
            api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
        )

        #TODO 修改system_prompt和正常prompt，使其可以通过用户profile+聊天记录合成新用户
        system_prompt = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with several user profiles and their chat history. Please strictly follow the requirements below, comprehensively consider these user profiles, and synthesize them into a new user profile.",

            "# INPUT EXAMPLE #":
                {"user_profile_list":[
                    {
                    "userName": "",  # Username
                    "bio": "",  # User biography
                    "behavior": []  # # User behavior history, format: 'item_id:item_title;item_descption;(genre_tags)'
                    }],
                "chat_history_list":[
                    {
                        "user_id": "",# as same as userName
                        "content":""# User's chat history.After the @ symbol, the corresponding ID is userName-1. For example, @1 actually @'s the user with userName 2.
                    }
                ]
                },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"Based on users' chat history,combine multiple users' \"bio\" to create a new \"bio\", aiming to maximally cover the original bio characteristics of multiple users.While trying to retain interests related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization."
                },
                {
                    "step": 3,
                    "description": f"For each user's \"behavior\", filter out items that are relatively irrelevant to the new user's characteristics and their chat history.While trying to retain items related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only item IDs and package them into a new JSON list as the new user's \"behavior\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",
                    "behavior": []  # User behavior history, format: 'item_id'
                }
            }
        }

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": origin_agent_info_list_str}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )

        return json.loads(response.choices[0].message.content)

    @staticmethod
    def get_merged_new_user_amazon(new_user_id,origin_agent_info_list_str,target_tag_list = []):
        origin_agent_info_list = json.loads(origin_agent_info_list_str)
        is_single = False
        if len(origin_agent_info_list) == 1:
            is_single = True

        client = OpenAI(
            base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
            api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
        )

        system_prompt = {
            "# CONTEXT #": "Assume you are a professional user profile synthesis expert. I will provide you with several user profiles. Please strictly follow the requirements below, comprehensively consider these user profiles, and synthesize them into a new user profile.",

            "# INPUT EXAMPLE #": {
                "userName": "",  # Username
                "bio": "",  # User biography
                "behavior": []  # User behavior history, format: 'item_id:item_title;item_descption;(genre_tags)'
            },

            "## SYNTHESIS PROCESS ##": [
                {
                    "step": 1,
                    "description": f"\"userName\" is fixed as \"{new_user_id}\""
                },
                {
                    "step": 2,
                    "description": f"Combine multiple users' \"bio\" to create a new \"bio\", aiming to maximally cover the original bio characteristics of multiple users.While trying to retain interests related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization."
                },
                {
                    "step": 3,
                    "description": f"For each user's \"behavior\", filter out items that are relatively irrelevant to the new user's characteristics.While trying to retain movies related to the following tags:[{target_tag_list}] as much as possible, also ensure generalization. Then extract only item IDs and package them into a new JSON list as the new user's \"behavior\""
                }
            ],

            "# RESPONSE #": {
                "description": "Please output in the following JSON format strictly:",
                "format": {
                    "userName": "",  # Username
                    "bio": "",
                    "behavior": []  # User behavior history, format: 'item_id'
                }
            }
        }

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": f"{system_prompt}"},
                {"role": "user", "content": origin_agent_info_list_str}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )

        return json.loads(response.choices[0].message.content)


    @staticmethod
    def iou_based_fusion(iou_matrix, target_groups=75):
        n_users = iou_matrix.shape[0]
        groups = [[i] for i in range(n_users)]
        group_representatives = list(range(n_users))  # 每个组的代表用户

        while len(groups) > target_groups:
            max_iou = -1
            best_pair = (-1, -1)

            # 寻找IOU最高的两个组
            for i in range(len(groups)):
                for j in range(i + 1, len(groups)):
                    # 使用组代表用户计算IOU
                    rep_i = group_representatives[i]
                    rep_j = group_representatives[j]
                    current_iou = iou_matrix[rep_i][rep_j]

                    if current_iou > max_iou:
                        max_iou = current_iou
                        best_pair = (i, j)

            if best_pair[0] != -1:
                i, j = best_pair
                # 合并组
                groups[i].extend(groups[j])
                groups.pop(j)

                # 更新组代表：选择与其他组平均IOU最高的用户作为新代表
                group_representatives.pop(j)
                best_rep = -1
                best_avg_iou = -1

                for user in groups[i]:
                    avg_iou = 0
                    count = 0
                    for other_group_idx, other_rep in enumerate(group_representatives):
                        if other_group_idx != i:
                            avg_iou += iou_matrix[user][other_rep]
                            count += 1
                    avg_iou = avg_iou / count if count > 0 else 0




if __name__ == '__main__':
    # # 读取 JSON 文件
    # # with open("C:\\Users\\lenovo\\PycharmProjects\\oasis\\generated_user_post_cosine_similarities\\user_post_cosine_similarities.json", "r") as f:
    # #     loaded_list = json.load(f)  # 读取为 Python 列表
    # #
    # # # 转换为 NumPy 数组
    # # user_matrix = np.array(loaded_list)
    # # print(f"user_matrix:\n{user_matrix}")
    # #
    # # iou_matrix = MergeAgentProcessor.calculate_iou_similarity(user_matrix)
    # # print(f"iou_matrix:\n{iou_matrix}")
    # # print(f"iou_matrix.shape:{iou_matrix.shape}")
    #
    # embbeding_file_path = 'C:\\Users\\lenovo\\PycharmProjects\\oasis\\embeddings\\embeddings.npz'
    # user_vector,post_vector = EmbbedingSaver.load_embbeding(embbeding_file_path)
    # user_similarity = MergeAgentProcessor.cosine_similarity_matrix(user_vector)
    # print(f"user_similarity: \n{user_similarity}")
    # post_similarity = MergeAgentProcessor.cosine_similarity_matrix(post_vector)
    # print(f"post_similarity: \n{post_similarity}")
    #
    # user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map()
    # user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)
    # user_behavior_similarity = MergeAgentProcessor.cosine_similarity_matrix(np.array(user_behavior_vector))
    # print(f"user_behavior_similarity:\n{user_behavior_similarity}")
    #
    # new_agent_id_and_origin_id_map = MergeAgentProcessor.kmeans_get_merged_new_agent_id_and_origin_id_map(user_behavior_vector)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")
    #
    # agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info("C:\\Users\\lenovo\\PycharmProjects\\oasis\\generated_user_profile\\user_100_profile.json")
    # print(f"agent_info_list:\n{agent_info_list}")
    #
    # generated_user_json_list = []
    # target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']
    # for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
    #     new_user_id = new_agent_id + 1
    #     origin_agent_info_list = []
    #     for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
    #         origin_agent_info_list.append(agent_info_list[origin_agent_id])
    #     origin_agent_info_list_str = json.dumps(origin_agent_info_list)
    #     result = MergeAgentProcessor.get_merged_new_user(new_user_id,origin_agent_info_list_str,target_tag_list)
    #     # 保证行为流一致
    #     for index in range(len(result['behavior'])):
    #         result['behavior'][index] = result['behavior'][index].split(":")[0]
    #     print(f"new_agent_id:{new_agent_id}")
    #     print(f"result:\n{result}")
    #     generated_user_json_list.append(result)
    # with open("C:\\Users\\lenovo\\PycharmProjects\\oasis\\generated_user_profile\\user_57_profile(1).json", "w") as file:
    #     json.dump(generated_user_json_list, file, indent=4)
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_1.json")[9],indent=4))
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_2.json")[9],indent=4))
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_3.json")[9],indent=4))
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_4.json")[9],indent=4))
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_5.json")[9],indent=4))
    print(json.dumps(MergeAgentProcessor.load_user_profile_with_extra_info("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_6.json")[9],indent=4))


