import asyncio
import json
import os

from camel.models import ModelFactory
from camel.types import ModelPlatformType, ModelType

import oasis
from oasis import (ActionType, LLMAction, ManualAction,
                   generate_twitter_agent_graph)
from tool.movie_len_loader import MovieLenLoader


async def main():
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

    profile_path = "../generated_user_profile/behavior_length_6/user_100_profile.json"
    agent_graph = await generate_twitter_agent_graph(
        profile_path=profile_path,
        model=openai_model,
        available_actions=available_actions,
    )

    # Define the path to the database
    db_path = "../data/twitter_simulation.db"

    # Delete the old database
    if os.path.exists(db_path):
        os.remove(db_path)

    # Make the environment
    env = oasis.make(
        agent_graph=agent_graph,
        platform=oasis.DefaultPlatformType.TWITTER,
        database_path=db_path,
        max_rec_len = 5
    )

    # Run the environment
    await env.reset()

    #电影信息
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    #用户行为流
    user_id_and_behavior_map = MovieLenLoader.get_user_id_and_behavior_map()

    max_agent_id = 99
    actions_1 = {}
    for agent_id in range(max_agent_id+1):
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

    #初始化用户行为流, TODO改为用json拿
    agent_info = ''
    with open(profile_path, "r", encoding='utf-8') as file:
        agent_info = json.load(file)
    for agent_id in range(max_agent_id+1):
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

    env.agent_graph.get_agent(0).memory.clear()

    actions_2 = {}  # 初始化空字典

    # 遍历所有 agent
    for _, agent in env.agent_graph.get_agents():
        actions_2[agent] = LLMAction()

    # Perform the actions
    await env.step(actions_2)



    # Close the environment
    await env.close()


if __name__ == "__main__":
    asyncio.run(main())