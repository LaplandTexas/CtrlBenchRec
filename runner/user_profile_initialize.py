import json
import random

import pandas as pd
from openai import OpenAI

from tool import movie_len_loader
from tool.amazon_loader import AmazonLoader
from tool.merge_agent_processor import MergeAgentProcessor


def movie_len_user(user_profile_path):
    movieLenLoader = movie_len_loader.MovieLenLoader()
    ratings,movies,users = movieLenLoader.load_movielens_data()

    rating_df = pd.DataFrame(ratings)
    movies_df = pd.DataFrame(movies)
    users_df = pd.DataFrame(users)

    # 职业编码映射
    occupation_mapping = {
        0: "other", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
        4: "college/grad student", 5: "customer service", 6: "doctor/health care",
        7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student",
        11: "lawyer", 12: "programmer", 13: "retired", 14: "sales/marketing",
        15: "scientist", 16: "self-employed", 17: "technician/engineer",
        18: "tradesman/craftsman", 19: "unemployed", 20: "writer"
    }

    # 年龄分段映射
    age_mapping = {
        1: "Under 18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+"
    }

    # 性别映射
    gender_mapping = {
        'F': "Female",
        'M': "Male"
    }
    users_df['occupation']=users_df['occupation'].map(occupation_mapping)
    users_df['age_during']=users_df['age'].map(age_mapping)
    users_df['gender']=users_df['gender'].map(gender_mapping)

    # 初始化客户端（假设 DeepSeek 的 API 兼容 OpenAI）
    client = OpenAI(
        base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
        api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
    )

    k = 100
    # 1. 获取前k个不同的userId
    top_k_users = rating_df['userId'].unique()[:k]

    # 2. 筛选出这k个用户的数据
    filtered_df = rating_df[rating_df['userId'].isin(top_k_users)]

    # 3. 按userId分组，并对每组按timestamp排序
    user_data_map = {}
    for user_id, group in filtered_df.groupby('userId'):
        # 对每个用户的数据按timestamp排序
        sorted_group = group.sort_values('timestamp')
        # 将排序后的DataFrame存入字典
        user_data_map[user_id] = sorted_group

    user_profile_list = []
    index = 1
    for user_id in user_data_map.keys():
        # 根据userId拿到user信息
        user_info = users_df[users_df['userId'] == user_id].iloc[0]  # 假设userId是唯一的
        gender = user_info['gender']
        age_during = user_info['age_during']
        user_info = user_info.drop(['age','zipcode'])
        user_info_json = user_info.to_json()
        # print(user_info_json)

        #拿到用户行为流,并和电影信息做拼接
        user_behavior = user_data_map.get(user_id)
        # user_behavior = user_behavior.head(10)
        merged_user_behavior = pd.merge(user_behavior,movies_df, on='movieId',how='left')
        merged_user_behavior.drop('timestamp',axis=1,inplace=True)
        merged_user_behavior_json = merged_user_behavior.to_json(orient='records')
        # print(merged_user_behavior_json)

        random_number = random.randint(1, 3)
        agent_behavioral_type = ''
        if random_number == 1:
            agent_behavioral_type = 'exploratory'
        if random_number == 2:
            agent_behavioral_type = 'focused'
        if random_number == 3:
            agent_behavioral_type = 'random'
        json_template = '''
        {
            "userName": "",#as same as userId,example:userId=1;userName=1
            "bio": "",
            "persona": "",
            "age": ,
            "gender": "",
            "mbti": "",
            "profession": "",
            "agent_behavioral_type":""# The behavioral style.Randomly assign one of these three types:"exploratory","focused","random"
            "interested_topics": [
                "",...
            ],
            "behavior":[
                "",...
            ]# Randomly choose 12 user's favorite movieId from user_behavior_json,remember you can only choose movieId!.
        }'''

        prompt = ("Assume you are a professional user profiling expert. "
                  "Below is the {user_info_json}, which represents the user's basic information,"
                  " and {user_behavior_json}, which represents the user's behavioral stream of watching movies and rating them.\n"
                  f"user_info_json:{user_info_json}\n"
                  f"user_behavior_json:{merged_user_behavior_json}\n"
                  f"agent_behavioral_type:{agent_behavioral_type}\n"
                  "Please synthesize a more detailed user profile based on these information, and return it in the following example JSON format:"
                  f"{json_template}")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert user profile generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )

        generated_profile = json.loads(response.choices[0].message.content)
        user_profile_list.append(generated_profile)
        print(generated_profile)

    with open(user_profile_path, "w") as file:
        json.dump(user_profile_list, file, indent=4)

def amazon_user():
    user_id_and_item_id_map = AmazonLoader.get_user_id_and_item_id_map()
    user_profile_list = []
    # 获取前100个用户
    for index in range(1,101):
        user_name = str(index)
        behavior = user_id_and_item_id_map[user_name]
        if len(behavior) > 6:
            behavior = behavior[:6]
        print(behavior)
        user_profile = {}
        user_profile['userName'] = user_name
        user_profile['bio'] = ""
        user_profile['behavior'] = behavior
        user_profile_list.append(user_profile)
    with open("..\\generated_user_profile\\behavior_length_6_amazon\\user_100_profile.json", "w") as file:
        json.dump(user_profile_list, file, indent=4)
    user_profile_list_with_extra_info = MergeAgentProcessor.load_user_profile_with_extra_info_amazon("..\\generated_user_profile\\behavior_length_6_amazon\\user_100_profile.json")
    generated_profile_list = []
    for user_profile in user_profile_list_with_extra_info:
        # 初始化客户端（假设 DeepSeek 的 API 兼容 OpenAI）
        client = OpenAI(
            base_url="https://api.deepseek.com/v1",  # 替换为实际 API 地址
            api_key="sk-71b2ae1a8c414ea6b8cc08f4f5888ca8"
        )

        prompt = f"""
        # Role
        You are an expert User Profiling Analyst.

        # Task
        Reverse-engineer a user's "Bio" based on their "behavior" stream and restructure the data.

        # Formatting Rules
        1. Bio Field: Synthesize a first-person bio under 40 words.
        2. Behavior Field: Keep ONLY the numerical IDs (strip all metadata).
        3. Output: Return ONLY the final JSON object.

        # Input Data
        {user_profile}

        # Expected JSON Output Structure
        {{
          "userName": "", # the same as input
          "bio": "[Generated Bio]",
          "behavior": ["ID1", "ID2"]
        }}
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are an expert user profile generator."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            response_format={"type": "json_object"}  # 强制 JSON 输出
        )

        generated_profile = json.loads(response.choices[0].message.content)
        generated_profile_list.append(generated_profile)
    with open("..\\generated_user_profile\\behavior_length_6_amazon\\user_100_profile.json", "w") as file:
        json.dump(generated_profile_list, file, indent=4)



if __name__ == '__main__':
    user_profile_path = "../generated_user_profile/behavior_length_6_sasrec/user_100_profile.json"
    movie_len_user(user_profile_path)




