import asyncio
import json
import math
import os
import sqlite3
import traceback
from datetime import datetime

import numpy as np
from camel.models import ModelFactory

from camel.types import ModelPlatformType, ModelType
from lazy_object_proxy.utils import await_
from numpy.f2py.auxfuncs import throw_error

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph, SocialAgent, DefaultPlatformType)
from tool import rec_history_processor
from tool.amazon_loader import AmazonLoader
from tool.embedding_processor import EmbeddingProcessor
from tool.merge_agent_processor import MergeAgentProcessor
from tool.movie_len_loader import MovieLenLoader
from tool.path_constants import PathConstants
from tool.rec_history_processor import RecHistoryProcessor
from tool.token_statistic_processor import TokenStatisticProcessor


async def user_rec_interact(profile_path,db_path,max_rec_len = 3):
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
    )

    # Define self model
    # openai_model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OLLAMA,
    #     model_type="qwen2.5-7b",
    # )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    # TODO 可在此修改推荐模型
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        max_rec_len = max_rec_len
    )

    # Run the environment
    await env.reset()

    #电影信息
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()

    user_profile_list = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        user_profile_list = json.load(f)
    actions_1 = {}
    for agent_id in range(len(user_profile_list)):
        actions_1[env.agent_graph.get_agent(agent_id)] = []

    #获取最大电影id
    max_movie_id = 3952
    #初始化电影帖子
    for movie_id in range(1,max_movie_id+1):
        content = ""
        movie_id = str(movie_id)
        if movie_id in movie_id_and_info_map:
            movie_info = movie_id_and_info_map.get(movie_id)
            movie_name = movie_info['title']
            movie_genres = movie_info['genres']
            content = (f"{movie_name},({movie_genres})")

        actions_1[env.agent_graph.get_agent(0)].append(
            ManualAction(action_type=ActionType.CREATE_POST,
                         action_args={
                             "content": f"{content}"})
        )

    agent_info = ''
    with open(profile_path, "r", encoding='utf-8') as file:
        agent_info = json.load(file)
    for agent_id in range(len(user_profile_list)):
        #agent_id从0开始，user_id从1开始
        user_name = agent_info[agent_id]['userName']
        # 初始化行为流
        agent_history_behaviors = agent_info[agent_id]['behavior']
        for i in range(len(agent_history_behaviors)):
            movie_id = int(agent_history_behaviors[i])
            actions_1[env.agent_graph.get_agent(agent_id)].append(
                ManualAction(action_type=ActionType.LIKE_POST,
                             action_args={
                                 "post_id": f"{movie_id}"
                             })
            )

    await env.step(actions_1)

    # 清除记忆防止时间过长
    env.agent_graph.get_agent(0).memory.clear()
    actions_2 = {}  # 初始化空字典

    # 遍历所有 agent
    for id, agent in env.agent_graph.get_agents():
        actions_2[agent] = LLMAction()
        # if agent.memory is not None:
        #     memory = agent.memory.retrieve()
        #     with open(f"../log/agent_{id}.json", 'w', encoding='utf-8') as f:
        #         json.dump(memory, f)

    # Perform the actions
    await env.step(actions_2)

    # Close the environment
    await env.close()

async def user_rec_interact_amazon(profile_path,db_path,max_rec_len = 3):
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
    )

    # Define self model
    # openai_model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OLLAMA,
    #     model_type="qwen2.5-7b",
    # )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    # TODO 可在此修改推荐模型
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        max_rec_len = max_rec_len
    )

    # Run the environment
    await env.reset()

    #商品信息
    item_id_and_info_map = AmazonLoader.get_item_id_and_info_map()

    user_profile_list = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        user_profile_list = json.load(f)
    actions_1 = {}
    for agent_id in range(len(user_profile_list)):
        actions_1[env.agent_graph.get_agent(agent_id)] = []

    #获取最大商品id
    max_item_id = 4234
    #初始化商品帖子
    for item_id in range(1,max_item_id+1):
        content = ""
        item_id = str(item_id)
        if item_id in item_id_and_info_map:
            item_info = item_id_and_info_map.get(item_id)
            item_title = item_info['Title']
            item_description = item_info['Description']
            item_category = item_info['Category']
            content = (f"title:{item_title}\ncatagory:{item_category}")

        actions_1[env.agent_graph.get_agent(0)].append(
            ManualAction(action_type=ActionType.CREATE_POST,
                         action_args={
                             "content": f"{content}"})
        )

    agent_info = ''
    with open(profile_path, "r", encoding='utf-8') as file:
        agent_info = json.load(file)
    for agent_id in range(len(user_profile_list)):
        #agent_id从0开始，user_id从1开始
        user_name = agent_info[agent_id]['userName']
        # 初始化行为流
        agent_history_behaviors = agent_info[agent_id]['behavior']
        for i in range(len(agent_history_behaviors)):
            item_id = int(agent_history_behaviors[i])
            actions_1[env.agent_graph.get_agent(agent_id)].append(
                ManualAction(action_type=ActionType.LIKE_POST,
                             action_args={
                                 "post_id": f"{item_id}"
                             })
            )

    await env.step(actions_1)


    # 清除记忆防止时间过长
    env.agent_graph.get_agent(0).memory.clear()
    actions_2 = {}  # 初始化空字典

    # 遍历所有 agent
    for id, agent in env.agent_graph.get_agents():
        actions_2[agent] = LLMAction()

    # Perform the actions
    await env.step(actions_2)

    # Close the environment
    await env.close()

async def user_rec_interact_with_detailed_behavior(profile_path,extra_info_path,db_path,max_rec_len = 3):
    # Define the model for the agents
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
    )

    # Define self model
    # openai_model = ModelFactory.create(
    #     model_platform=ModelPlatformType.OLLAMA,
    #     model_type="qwen2.5-7b",
    # )

    # Define the available actions for the agents
    available_actions = [
        ActionType.LIKE_POST
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=extra_info_path,
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    # TODO 可在此修改推荐模型
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        max_rec_len = max_rec_len
    )

    # Run the environment
    await env.reset()

    #电影信息
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()

    user_profile_list = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        user_profile_list = json.load(f)
    actions_1 = {}
    actions_0 = {}
    for agent_id in range(len(user_profile_list)):
        actions_1[env.agent_graph.get_agent(agent_id)] = []
        actions_0[env.agent_graph.get_agent(agent_id)] = []

    #获取最大电影id
    max_movie_id = 3952
    #初始化电影帖子
    for movie_id in range(1,max_movie_id+1):
        content = ""
        movie_id = str(movie_id)
        if movie_id in movie_id_and_info_map:
            movie_info = movie_id_and_info_map.get(movie_id)
            movie_name = movie_info['title']
            movie_genres = movie_info['genres']
            content = (f"{movie_name},({movie_genres})")

        actions_0[env.agent_graph.get_agent(0)].append(
            ManualAction(action_type=ActionType.CREATE_POST,
                         action_args={
                             "content": f"{content}"})
        )

    await env.step(actions_0)

    env.agent_graph.get_agent(0).memory.clear()

    agent_info = ''
    with open(profile_path, "r", encoding='utf-8') as file:
        agent_info = json.load(file)
    for agent_id in range(len(user_profile_list)):
        #agent_id从0开始，user_id从1开始
        user_name = agent_info[agent_id]['userName']
        # 初始化行为流
        agent_history_behaviors = agent_info[agent_id]['behavior']
        for i in range(len(agent_history_behaviors)):
            movie_id = int(agent_history_behaviors[i])
            actions_1[env.agent_graph.get_agent(agent_id)].append(
                ManualAction(action_type=ActionType.LIKE_POST,
                             action_args={
                                 "post_id": f"{movie_id}"
                             })
            )

    await env.step(actions_1)

    actions_2 = {}  # 初始化空字典

    # 遍历所有 agent
    for id, agent in env.agent_graph.get_agents():
        actions_2[agent] = LLMAction()

    # Perform the actions
    await env.step(actions_2)

    print("memory:",env.agent_graph.get_agent(1).memory.retrieve())

    # Close the environment
    await env.close()

async def user_group_debate(profile_path,db_path,new_agent_id_and_origin_id_map):
    openai_model = ModelFactory.create(
        model_platform=ModelPlatformType.DEEPSEEK,
        model_type=ModelType.DEEPSEEK_CHAT,
    )

    # Define the available actions for the agents
    available_actions = [
        ActionType.SEND_TO_GROUP
    ]

    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=openai_model,
        available_actions=available_actions,
    )


    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    # TODO 可在此修改推荐模型
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path
    )

    # Run the environment
    await env.reset()

    # 初始化群组
    actions_1 = {}
    group_id_set = set()

    for group_id in range(len(new_agent_id_and_origin_id_map)):
        group_id = str(group_id)
        origin_agent_id_list = new_agent_id_and_origin_id_map[group_id]
        for agent_id in origin_agent_id_list:
            if group_id not in group_id_set:
                group_result = await env.platform.create_group(agent_id, "")
                db_group_id = group_result["group_id"]
                group_id_set.add(group_id)
            else:
                actions_1[env.agent_graph.get_agent(agent_id)] = ManualAction(
                    action_type=ActionType.JOIN_GROUP, action_args={"group_id": db_group_id})

    await env.step(actions_1)

    # agent自主讨论
    actions_2 = {}

    # 遍历所有 agent
    for id, agent in env.agent_graph.get_agents():
        actions_2[agent] = LLMAction()

    # Perform the actions
    await env.step(actions_2)

    # agent自主讨论
    actions_3 = {}

    # 遍历所有 agent
    for id, agent in env.agent_graph.get_agents():
        actions_3[agent] = LLMAction()
    # Perform the actions
    await env.step(actions_3)

    # Close the environment
    await env.close()

def at_the_end_of_small_epoch(profile_path, db_path,current_big_epoch,current_small_epoch,profile_prefix):
    print(f"{datetime.now()}.Begin:at_the_end_of_small_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"

    agent_info_list = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        agent_info_list = json.load(f)

    for agent_id in range(len(agent_info_list)):
        # 数据库里的user_id和agent_id相同
        user_id = agent_id
        agent_info_list[agent_id]['behavior'] = user_id_and_like_map.get(user_id)

    json.dump(agent_info_list, open(new_profile_path, 'w', encoding='utf-8'),indent=4)

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(new_profile_path)

    generated_user_json_list = []
    target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']
    for agent_id in range(len(agent_info_list)):
        # profile里的user_id为agent_id+1
        user_id = agent_id + 1
        origin_agent_info_list = []
        origin_agent_info_list.append(agent_info_list[agent_id])
        origin_agent_info_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user(user_id,origin_agent_info_str,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        generated_user_json_list.append(result)

    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    print(f"{datetime.now()}.End:at_the_end_of_small_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")


def at_the_end_of_small_epoch_without_reflect(profile_path, db_path,current_big_epoch,current_small_epoch,profile_prefix):
    # 统计token消耗量
    TokenStatisticProcessor.dump_to_json(profile_prefix + f"_token_{current_big_epoch}_{current_small_epoch}.json")
    print(f"{datetime.now()}.Begin:at_the_end_of_small_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_id_and_rec_map = MergeAgentProcessor.get_user_id_and_rec_map(db_path)
    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    new_rec_history = profile_prefix + f"_rec_history_{current_big_epoch}_{current_small_epoch}.json"

    agent_info_list = []
    rec_history_list = []
    with open(profile_path, 'r', encoding='utf-8') as f:
        agent_info_list = json.load(f)
    if os.path.exists(PathConstants.total_rec_history_path):
        with open (PathConstants.total_rec_history_path,'r',encoding='utf-8') as f:
            rec_history_list = json.load(f)
    else:
        for agent_id in range(len(agent_info_list)):
            rec_history_list.append([])

    for agent_id in range(len(agent_info_list)):
        # 数据库里的user_id和agent_id相同
        user_id = agent_id
        agent_info_list[agent_id]['behavior'] = user_id_and_like_map.get(user_id)
        rec_history = user_id_and_rec_map.get(user_id)
        rec_history_list[agent_id] = rec_history_list[agent_id] + rec_history


    json.dump(agent_info_list, open(new_profile_path, 'w', encoding='utf-8'),indent=4)
    json.dump(rec_history_list, open(new_rec_history, 'w', encoding='utf-8'),indent=4)
    json.dump(rec_history_list, open(PathConstants.total_rec_history_path, 'w', encoding='utf-8'),indent=4)

def at_the_end_of_big_epoch(profile_path, db_path,current_big_epoch,current_small_epoch,profile_prefix):
    # 清除历史token数据
    TokenStatisticProcessor.clear()
    print(f"{datetime.now()}.Begin:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    embbeding_file_path = PathConstants.embedding_storage_file_path
    user_vector,post_vector = EmbeddingProcessor.load_embedding(embbeding_file_path)
    user_similarity = MergeAgentProcessor.cosine_similarity_matrix(user_vector)
    # print(f"user_similarity: \n{user_similarity}")
    post_similarity = MergeAgentProcessor.cosine_similarity_matrix(post_vector)
    # print(f"post_similarity: \n{post_similarity}")

    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)
    user_behavior_similarity = MergeAgentProcessor.cosine_similarity_matrix(np.array(user_behavior_vector))
    # print(f"user_behavior_similarity:\n{user_behavior_similarity}")

    new_agent_id_and_origin_id_map = MergeAgentProcessor.kmeans_get_merged_new_agent_id_and_origin_id_map(user_behavior_vector)
    k_means_path = profile_prefix + f"_k_means_{current_big_epoch}_{current_small_epoch}.json"
    with open(k_means_path, 'w', encoding='utf-8') as f:
        json.dump(new_agent_id_and_origin_id_map, f, indent=4)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(profile_path)
    # print(f"agent_info_list:\n{agent_info_list}")

    generated_user_json_list = []
    target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user(new_user_id,origin_agent_info_list_str,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    # 统计token消耗量
    TokenStatisticProcessor.dump_to_json(
        profile_prefix + f"_token_{current_big_epoch}_{current_small_epoch}.json" )
    print(f"{datetime.now()}.End:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")

async def at_the_end_of_big_epoch_with_debate(profile_path, db_path, current_big_epoch, current_small_epoch, profile_prefix):
    # 清除历史token数据
    TokenStatisticProcessor.clear()
    print(f"{datetime.now()}.Begin:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    embbeding_file_path = PathConstants.embedding_storage_file_path
    user_vector,post_vector = EmbeddingProcessor.load_embedding(embbeding_file_path)

    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)

    new_agent_id_and_origin_id_map = MergeAgentProcessor.kmeans_get_merged_new_agent_id_and_origin_id_map(user_behavior_vector)
    k_means_path = profile_prefix + f"_k_means_{current_big_epoch}_{current_small_epoch}.json"
    with open(k_means_path, 'w', encoding='utf-8') as f:
        json.dump(new_agent_id_and_origin_id_map, f, indent=4)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(profile_path)
    # print(f"agent_info_list:\n{agent_info_list}")

    try:
        await user_group_debate(profile_path,db_path,new_agent_id_and_origin_id_map)
    except Exception as e:
        traceback.print_exc()
        raise


    chat_history_path = profile_prefix + f"_chat_history_{current_big_epoch}_{current_small_epoch}.json"
    with open(chat_history_path, 'w', encoding='utf-8') as f:
        json.dump(MergeAgentProcessor.get_group_id_and_chat_history_map(),f,indent=4)

    generated_user_json_list = []
    target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user_with_chat_history(new_user_id,origin_agent_info_list_str,chat_history_path,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    # 统计token消耗量
    TokenStatisticProcessor.dump_to_json(
        profile_prefix + f"_token_{current_big_epoch}_{current_small_epoch}.json" )
    print(f"{datetime.now()}.End:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")

async def at_the_end_of_big_epoch_with_debate_amazon(profile_path, db_path, current_big_epoch, current_small_epoch, profile_prefix):
    # 清除历史token数据
    TokenStatisticProcessor.clear()
    print(f"{datetime.now()}.Begin:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    embbeding_file_path = PathConstants.embedding_storage_file_path
    user_vector,post_vector = EmbeddingProcessor.load_embedding(embbeding_file_path)
    user_similarity = MergeAgentProcessor.cosine_similarity_matrix(user_vector)
    # print(f"user_similarity: \n{user_similarity}")
    post_similarity = MergeAgentProcessor.cosine_similarity_matrix(post_vector)
    # print(f"post_similarity: \n{post_similarity}")

    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)
    user_behavior_similarity = MergeAgentProcessor.cosine_similarity_matrix(np.array(user_behavior_vector))
    # print(f"user_behavior_similarity:\n{user_behavior_similarity}")

    new_agent_id_and_origin_id_map = MergeAgentProcessor.kmeans_get_merged_new_agent_id_and_origin_id_map(user_behavior_vector)
    k_means_path = profile_prefix + f"_k_means_{current_big_epoch}_{current_small_epoch}.json"
    with open(k_means_path, 'w', encoding='utf-8') as f:
        json.dump(new_agent_id_and_origin_id_map, f, indent=4)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info_amazon(profile_path)
    # print(f"agent_info_list:\n{agent_info_list}")

    try:
        await user_group_debate(profile_path,db_path,new_agent_id_and_origin_id_map)
    except Exception as e:
        traceback.print_exc()
        raise


    chat_history_path = profile_prefix + f"_chat_history_{current_big_epoch}_{current_small_epoch}.json"
    with open(chat_history_path, 'w', encoding='utf-8') as f:
        json.dump(MergeAgentProcessor.get_group_id_and_chat_history_map(),f,indent=4)

    generated_user_json_list = []
    target_tag_list= ["Action Figures & Statues","Arts & Crafts","Puzzles","Drawing & Painting Supplies","Science"]
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user_with_chat_history_amazon(new_user_id,origin_agent_info_list_str,chat_history_path,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    # 统计token消耗量
    TokenStatisticProcessor.dump_to_json(
        profile_prefix + f"_token_{current_big_epoch}_{current_small_epoch}.json" )
    print(f"{datetime.now()}.End:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")


def at_the_end_of_big_epoch_amazon(profile_path, db_path,current_big_epoch,current_small_epoch,profile_prefix):
    # 清除历史token数据
    TokenStatisticProcessor.clear()
    print(f"{datetime.now()}.Begin:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    embbeding_file_path = PathConstants.embedding_storage_file_path
    user_vector,post_vector = EmbeddingProcessor.load_embedding(embbeding_file_path)

    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)
    user_behavior_similarity = MergeAgentProcessor.cosine_similarity_matrix(np.array(user_behavior_vector))
    # print(f"user_behavior_similarity:\n{user_behavior_similarity}")

    new_agent_id_and_origin_id_map = MergeAgentProcessor.kmeans_get_merged_new_agent_id_and_origin_id_map(user_behavior_vector)
    k_means_path = profile_prefix + f"_k_means_{current_big_epoch}_{current_small_epoch}.json"
    with open(k_means_path, 'w', encoding='utf-8') as f:
        json.dump(new_agent_id_and_origin_id_map, f, indent=4)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info_amazon(profile_path)
    # print(f"agent_info_list:\n{agent_info_list}")

    generated_user_json_list = []
    # TODO对amazon数据集进行分析，定义目标标签
    target_tag_list = ["Action Figures & Statues", "Arts & Crafts", "Puzzles", "Drawing & Painting Supplies", "Science"]
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user_amazon(new_user_id,origin_agent_info_list_str,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    # 统计token消耗量
    TokenStatisticProcessor.dump_to_json(
        profile_prefix + f"_token_{current_big_epoch}_{current_small_epoch}.json" )
    print(f"{datetime.now()}.End:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")

def at_the_end_of_big_epoch_with_gmm(profile_path, db_path,current_big_epoch,current_small_epoch,profile_prefix):
    print(f"{datetime.now()}.Begin:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")
    embbeding_file_path = PathConstants.embedding_storage_file_path
    user_vector,post_vector = EmbeddingProcessor.load_embedding(embbeding_file_path)
    user_similarity = MergeAgentProcessor.cosine_similarity_matrix(user_vector)
    # print(f"user_similarity: \n{user_similarity}")
    post_similarity = MergeAgentProcessor.cosine_similarity_matrix(post_vector)
    # print(f"post_similarity: \n{post_similarity}")

    user_id_and_like_map = MergeAgentProcessor.get_user_id_and_like_map(db_path)
    user_behavior_vector = MergeAgentProcessor.get_user_behavior_vector(user_id_and_like_map,user_vector,post_vector)
    user_behavior_similarity = MergeAgentProcessor.cosine_similarity_matrix(np.array(user_behavior_vector))
    # print(f"user_behavior_similarity:\n{user_behavior_similarity}")

    new_agent_id_and_origin_id_map,best_k = MergeAgentProcessor.gmm_get_agent_cluster_probabilities(user_behavior_vector)
    k_means_path = profile_prefix + f"_gmm_{current_big_epoch}_{current_small_epoch}.json"
    with open(k_means_path, 'w', encoding='utf-8') as f:
        json.dump(new_agent_id_and_origin_id_map, f, indent=4)
    # print(f"new_agent_id_and_origin_id_map:\n{new_agent_id_and_origin_id_map}")

    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(profile_path)
    # print(f"agent_info_list:\n{agent_info_list}")

    generated_user_json_list = []
    target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[new_agent_id]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user(new_user_id,origin_agent_info_list_str,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    new_profile_path = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
    with open(new_profile_path, "w") as file:
        json.dump(generated_user_json_list, file, indent=4)

    print(f"{datetime.now()}.End:at_the_end_of_big_epoch,current_big_epoch:{current_big_epoch},current_small_epoch:{current_small_epoch}")


async def gmm_test():

    max_big_epoch = 1
    max_small_epoch = 1

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()

    last_user_profile = "../generated_user_profile/test_gmm/user_100_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/test_gmm/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch_with_gmm(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


async def test_bge():

    max_big_epoch = 1
    max_small_epoch = 20

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_rec_history_0_54.json")

    last_user_profile = "../generated_user_profile/bge_result/user_100_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/bge_result/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
            except Exception as e:
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def amazon():

    max_big_epoch = 1
    max_small_epoch = 40

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_amazon/test_1/user_28_profile_rec_history_0_33.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_amazon/test_1/user_28_profile_0_33.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_amazon/test_1/user_28_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 34

        # TODO 合并时，清除当前历史推荐记录；继续跑实验时 不清除
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact_amazon(last_user_profile, db_path,3)
            except Exception as e:
                traceback.print_exc()
                raise
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch_amazon(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def amazon_base():

    max_big_epoch = 1
    max_small_epoch = 40

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_amazon/base/user_100_profile_rec_history_0_26.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_amazon/base/user_100_profile_0_26.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_amazon/base/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 27

        # TODO 合并时，清除当前历史推荐记录；继续跑实验时 不清除
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact_amazon(last_user_profile, db_path,3)
            except Exception as e:
                traceback.print_exc()
                raise
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch_amazon(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def test_debate_amazon():

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile_0_1.json"
    db_path = "../data/twitter_simulation.db"

    k_means_path = "../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile_k_means_1_2.json"
    with open(k_means_path, 'r', encoding='utf-8') as f:
        new_agent_id_and_origin_id_map = json.load(f)
    try:
        await user_group_debate(last_user_profile,db_path,new_agent_id_and_origin_id_map)
    except Exception as e:
        traceback.print_exc()
        raise

    chat_history_path = "../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile_group_id_and_chat_history_1_2.json"
    with open(chat_history_path, 'w', encoding='utf-8') as f:
        json.dump(MergeAgentProcessor.get_group_id_and_chat_history_map(),f,indent=4)

    # agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(last_user_profile)
    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info_amazon(last_user_profile)
    # print(f"agent_info_list:\n{agent_info_list}")


    generated_user_json_list = []
    # target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    target_tag_list = ["Action Figures & Statues", "Arts & Crafts", "Puzzles", "Drawing & Painting Supplies", "Science"]
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[str(new_agent_id)]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user_with_chat_history_amazon(new_user_id, origin_agent_info_list_str, chat_history_path,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    print(generated_user_json_list)
    with open("../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile_1_2.json", "w", encoding='utf-8') as f:
        json.dump(generated_user_json_list,f,indent=4)

    print("done")

async def test_debate_ml1m():

    last_user_profile = "../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile_1_1.json"
    db_path = "../data/twitter_simulation.db"

    k_means_path = "../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile_k_means_1_2.json"
    with open(k_means_path, 'r', encoding='utf-8') as f:
        new_agent_id_and_origin_id_map = json.load(f)
    try:
        await user_group_debate(last_user_profile,db_path,new_agent_id_and_origin_id_map)
    except Exception as e:
        traceback.print_exc()
        raise

    chat_history_path = ("../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile_group_id_and_chat_history_1_2.json")
    with open(chat_history_path, 'w', encoding='utf-8') as f:
        json.dump(MergeAgentProcessor.get_group_id_and_chat_history_map(),f,indent=4)

    # agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(last_user_profile)
    agent_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(last_user_profile)
    # print(f"agent_info_list:\n{agent_info_list}")


    generated_user_json_list = []
    target_tag_list = ['Action', 'Adventure', 'Thriller', 'War', 'Romance', 'Film-Noir']
    for new_agent_id in range(len(new_agent_id_and_origin_id_map.keys())):
        new_user_id = new_agent_id + 1
        origin_agent_info_list = []
        for origin_agent_id in new_agent_id_and_origin_id_map[str(new_agent_id)]:
            origin_agent_info_list.append(agent_info_list[origin_agent_id])
        origin_agent_info_list_str = json.dumps(origin_agent_info_list)
        result = MergeAgentProcessor.get_merged_new_user_with_chat_history(new_user_id, origin_agent_info_list_str, chat_history_path,target_tag_list)
        # 保证行为流一致
        for index in range(len(result['behavior'])):
            result['behavior'][index] = result['behavior'][index].split(":")[0]
        print(f"new_agent_id:{new_agent_id}")
        print(f"result:\n{result}")
        generated_user_json_list.append(result)

    print(generated_user_json_list)
    with open("../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile_1_2.json", "w", encoding='utf-8') as f:
        json.dump(generated_user_json_list,f,indent=4)

    print("done")

async def main():

    max_big_epoch = 1
    max_small_epoch = 200

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_rec_history_0_182.json")

    last_user_profile = "../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_0_182.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 183

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def debate_raw_main():

    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 1

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_rec_history_0_182.json")

    last_user_profile = "../generated_user_profile/behavior_length_6/test_debate_raw/user_100_profile_0_2.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6/test_debate_raw/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def debate_epoch_200():

    max_big_epoch = 1
    max_small_epoch = 20

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/debate/user_28_profile_rec_history_0_105.json")

    last_user_profile = "../generated_user_profile/behavior_length_6/debate/user_28_profile_0_1.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6/debate/user_28_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 2

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def debate_raw_main_amazon():

    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 1

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/epoch200_base/user_100_profile_rec_history_0_182.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile_0_2.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/amazon_raw/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact_amazon(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def task2(last_epoch_num):

    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    if last_epoch_num >= 0:
        # RecHistoryProcessor.load_rec_history(f"../generated_user_profile/behavior_length_6/debate/user_28_profile_rec_history_0_{last_epoch_num}.json")
        last_user_profile = f"../generated_user_profile/task2/l2/user_1_profile_0_{last_epoch_num}.json"
    else:
        last_user_profile = f"../generated_user_profile/task2/l2/user_1_profile.json"

    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6/debate/user_28_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = last_epoch_num+1

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                # agent_extra_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(last_user_profile)
                # last_user_profile_extra = f"{profile_prefix}_extra_info_{current_big_epoch}_{current_small_epoch}.json"
                # with open(last_user_profile_extra, "w", encoding="utf-8") as f:
                #     json.dump(agent_extra_info_list, f,indent=4)
                # await user_rec_interact_with_detailed_behavior(last_user_profile,last_user_profile_extra, db_path,3)
                await user_rec_interact(last_user_profile, db_path, 3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def task2_l3(last_epoch_num):

    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    if last_epoch_num >= 0:
        RecHistoryProcessor.load_rec_history(f"..generated_user_profile/task2/l3/user_28_profile_rec_history_0_{last_epoch_num}.json")
        last_user_profile = f"../generated_user_profile/task2/l3/user_28_profile_0_{last_epoch_num}.json"
    else:
        last_user_profile = f"../generated_user_profile/task2/l3/user_28_profile.json"

    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/task2/l3/user_28_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = last_epoch_num+1

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                # agent_extra_info_list = MergeAgentProcessor.load_user_profile_with_extra_info(last_user_profile)
                # last_user_profile_extra = f"{profile_prefix}_extra_info_{current_big_epoch}_{current_small_epoch}.json"
                # with open(last_user_profile_extra, "w", encoding="utf-8") as f:
                #     json.dump(agent_extra_info_list, f,indent=4)
                # await user_rec_interact_with_detailed_behavior(last_user_profile,last_user_profile_extra, db_path,3)
                await user_rec_interact(last_user_profile, db_path, 3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def task3():

    max_big_epoch = 1
    max_small_epoch = 40

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6/debate/user_28_profile_rec_history_0_105.json")

    last_user_profile = "../generated_user_profile/task2/task3_war/user_28_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/task2/task3_war/user_28_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def bge_merge():


    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 1

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/ml1m_raw/user_100_profile_rec_history_1_0.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/m1lm_raw_v1/user_100_profile_1_0.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/m1lm_raw_v1/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 1

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


async def bge_merge_k_means():


    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/ml1m_raw/user_100_profile_rec_history_1_0.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/m1lm_raw_v1_k_means/user_100_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/m1lm_raw_v1_k_means/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


async def bge_k_means_epoch_30():


    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/ml1m_raw/user_100_profile_rec_history_1_0.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_k_means/user_29_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_k_means/user_29_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0

        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def bge_ml1m_epoch30():
    max_big_epoch = 1
    max_small_epoch = 20

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile_rec_history_0_16.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile_0_16.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 17
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def bge_ml1m_epoch30():
    max_big_epoch = 1
    max_small_epoch = 20

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile_rec_history_0_16.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile_0_16.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/epoch30_ml1m_v2/user_29_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 17
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def bge_amazon_epoch30_base():
    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_0_6.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 7
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact_amazon(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def bge_amazon_epoch30():
    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_bge/epoch30_amazon/user_27_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_bge/epoch30_amazon/user_27_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact_amazon(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def narm_ml1m_merge():
    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 1

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_narm/ml1m_raw/user_100_profile_0_2.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_narm/ml1m_raw/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def narm_ml1m_merge_k_means():
    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 1

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_narm/ml1m_k_means_epoch30/user_100_profile_0_2.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_narm/ml1m_k_means_epoch30/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile, db_path, current_big_epoch, current_small_epoch, profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def narm_ml1m_k_means_epoch30():
    max_big_epoch = 1
    max_small_epoch = 30

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_narm/ml1m_k_means_epoch30/user_29_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_narm/ml1m_k_means_epoch30/user_29_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        at_the_end_of_big_epoch(last_user_profile, db_path, current_big_epoch, current_small_epoch, profile_prefix)
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")



async def narm_ml1m_debate_epoch20():
    max_big_epoch = 1
    max_small_epoch = 20

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_narm/ml1m_debate1_epoch20/user_31_profile_rec_history_0_9.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_narm/ml1m_debate1_epoch20/user_31_profile_0_9.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_narm/ml1m_debate1_epoch20/user_31_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 10
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path, 20)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def narm_ml1m_base_epoch20():
    max_big_epoch = 1
    max_small_epoch = 50

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_narm/ml1m_base_epoch20/user_100_profile_rec_history_0_15.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_narm/ml1m_base_epoch20/user_100_profile_0_15.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_narm/ml1m_base_epoch20/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 16
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path, 20)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


async def sasrec_ml1m_base_epoch20():
    max_big_epoch = 1
    max_small_epoch = 50

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_rec_history_0_12.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_0_12.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 13
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path, 3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")

async def sasrec_ml1m_debate_epoch20():
    max_big_epoch = 1
    max_small_epoch = 50

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_sasrec/ml1m_base_epoch20/user_100_profile_rec_history_0_12.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_sasrec/ml1m_debate_epoch20/user_27_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path, 3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile, db_path, current_big_epoch,
                                                      current_small_epoch, profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


async def sasrec_ml1m_merge():
    max_big_epoch = 2
    max_small_epoch = 2

    current_big_epoch = 0

    # TODO 初始时修改路径

    RecHistoryProcessor.clean()
    # RecHistoryProcessor.load_rec_history("../generated_user_profile/behavior_length_6_bge/epoch30_amazon_base/user_100_profile_rec_history_0_6.json")

    last_user_profile = "../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile.json"
    db_path = "../data/twitter_simulation.db"

    profile_prefix = "../generated_user_profile/behavior_length_6_sasrec/ml1m_raw_debate/user_100_profile"

    while current_big_epoch < max_big_epoch:

        current_small_epoch = 0
        # 大轮开始时，清除历史推荐记录
        # TODO 当需要继续跑实验时，不清除当前历史推荐记录
        # RecHistoryProcessor.clean()

        while current_small_epoch < max_small_epoch:
            # 小轮开始时，清除token数据
            TokenStatisticProcessor.clear()
            try:
                # 第一轮时推荐1个，第二轮时推荐2个，第三轮时推荐4个，以此类推
                await user_rec_interact(last_user_profile, db_path,3)
                # await user_rec_interact(last_user_profile, db_path,current_small_epoch+1)
                # await user_rec_interact(last_user_profile, db_path,min(pow(2,current_small_epoch),64))
            except Exception as e:
                print(e)
                pass
            # at_the_end_of_small_epoch(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            at_the_end_of_small_epoch_without_reflect(last_user_profile, db_path,current_big_epoch,current_small_epoch,profile_prefix)
            last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
            current_small_epoch = current_small_epoch + 1

        try:
            await at_the_end_of_big_epoch_with_debate(last_user_profile,db_path,current_big_epoch,current_small_epoch,profile_prefix)
        except Exception as e:
            print(traceback.print_exc())
        last_user_profile = profile_prefix + f"_{current_big_epoch}_{current_small_epoch}.json"
        current_big_epoch = current_big_epoch + 1

    print("done")


if __name__ == '__main__':
    # asyncio.run(test_bge())
    # asyncio.run(amazon())
    # asyncio.run(amazon_base())
    # asyncio.run(task2(19))
    # asyncio.run(main())
    # asyncio.run(debate_raw_main_amazon())
    asyncio.run(sasrec_ml1m_debate_epoch20())
    # asyncio.run(test_debate_amazon())
    # asyncio.run(debate_epoch_200())
    # asyncio.run(task3())
    # asyncio.run(sasrec_ml1m_merge())
    # asyncio.run(bge_ml1m_epoch40())
    # asyncio.run(bge_ml1m_epoch30())
    # asyncio.run(bge_amazon_epoch30_base())
    # asyncio.run(bge_amazon_epoch30())
    # asyncio.run(narm_ml1m_merge())
    # asyncio.run(narm_ml1m_debate_epoch20())
    # asyncio.run(narm_ml1m_base_epoch20())
    # asyncio.run(task2_l3(-1))
    # asyncio.run(debate_epoch_200())