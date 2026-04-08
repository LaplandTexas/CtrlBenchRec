import json

from tool.movie_len_loader import MovieLenLoader


def user_behavior_analysis(profile_path):
    print("Begin:user_behavior_analysis,profile_path={}".format(profile_path))
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    with open(profile_path, 'r', encoding='utf-8') as file:
        user_profile_list = json.load(file)
    for user_profile in user_profile_list:
        target_movie_set = set()
        genre_count_map = {}
        behaviors = user_profile['behavior']
        for behavior in behaviors:
            if behavior in target_movie_set:
                continue
            try:
                movie_info = movie_id_and_info_map[str(behavior)]
            except Exception as e:
                continue
            genres = movie_info['genres'].split('|')
            # print(f"{user_profile['userName']},{behavior},{genres}")
            for genre in genres:
                if genre in target_tag_list:
                    target_movie_set.add(behavior)
                    if genre not in genre_count_map:
                        genre_count_map[genre] = 1
                    else :
                        genre_count_map[genre] += 1
        for genre in genre_count_map:
            genre_count_map[genre] = round(genre_count_map[genre] / len(target_movie_set), 2)
        print(f"{user_profile['userName']},{len(target_movie_set)},{genre_count_map}")

def user_behavior_analysis_distinct(profile_path,raw_profile_path):
    print("Begin:user_behavior_analysis,profile_path={}".format(profile_path))
    movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
    with open(profile_path, 'r', encoding='utf-8') as file:
        user_profile_list = json.load(file)
    with open(raw_profile_path, 'r', encoding='utf-8') as file:
        raw_user_profile_list = json.load(file)
    for index in range(len(user_profile_list)):
        user_profile = user_profile_list[index]
        raw_user_profile = raw_user_profile_list[index]
        target_movie_set = set()
        genre_count_map = {}
        behaviors = []
        for behavior in user_profile['behavior']:
            if int(behavior) not in raw_user_profile['behavior']:
                behaviors.append(behavior)
        for behavior in behaviors:
            if behavior in target_movie_set:
                continue
            try:
                movie_info = movie_id_and_info_map[str(behavior)]
            except Exception as e:
                continue
            genres = movie_info['genres'].split('|')
            # print(f"{user_profile['userName']},{behavior},{genres}")
            for genre in genres:
                if genre in target_tag_list:
                    target_movie_set.add(behavior)
                    if genre not in genre_count_map:
                        genre_count_map[genre] = 1
                    else :
                        genre_count_map[genre] += 1
        for genre in genre_count_map:
            genre_count_map[genre] = round(genre_count_map[genre] / len(target_movie_set), 2)
        print(f"{user_profile['userName']},{len(target_movie_set)},{genre_count_map}")

if __name__ == '__main__':
    target_tag_list= ['Action','Adventure','Thriller','War','Romance','Film-Noir']

    # user_behavior_analysis("../generated_user_profile/task2/l1_change_profile_and_prompt/user_1_profile.json")
    # # 2, 6, {'Action': 0.33, 'Thriller': 1.0, 'Adventure': 0.17, 'Romance': 0.33}
    # user_behavior_analysis("../generated_user_profile/task2/l1_change_profile_and_prompt/user_1_profile_0_19.json")
    # # 2, 17, {'Action': 0.65, 'Thriller': 0.71, 'Adventure': 0.35, 'Romance': 0.18, 'War': 0.06}
    # user_behavior_analysis("../generated_user_profile/task2/l1_change_profile/user_1_profile_0_19.json")
    # # 2, 28,{'Action': 0.71, 'Thriller': 0.71, 'Adventure': 0.36, 'Romance': 0.14, 'War': 0.11}
    # user_behavior_analysis("../generated_user_profile/task2/l1_change_prompt/user_1_profile.json")
    # # 2,8,{'Action': 0.38, 'Adventure': 0.25, 'Thriller': 0.75, 'War': 0.12, 'Romance': 0.25}
    # user_behavior_analysis("../generated_user_profile/task2/l1_change_prompt/user_1_profile_0_19.json")
    # # 2,27,{'Action': 0.7, 'Adventure': 0.52, 'Thriller': 0.41, 'War': 0.19, 'Romance': 0.15}
    #
    # user_behavior_analysis_distinct("../generated_user_profile/task2/l1_change_profile_and_prompt/user_1_profile_0_19.json",
    #                                 "../generated_user_profile/task2/l1_change_profile_and_prompt/user_1_profile.json")

    #
    # user_behavior_analysis("../generated_user_profile/task2/l3/user_28_profile_0_29.json")
    # 10, 21, {'Romance': 0.19, 'Thriller': 0.38, 'War': 0.29, 'Action': 0.52, 'Adventure': 0.52}
    # user_behavior_analysis("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_5.json")
    # 10, 29, {'Romance': 0.17, 'Thriller': 0.38, 'War': 0.24, 'Action': 0.59, 'Adventure': 0.41}
    # user_behavior_analysis("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_10.json")
    # 10, 36, {'Romance': 0.19, 'Thriller': 0.42, 'War': 0.19, 'Action': 0.53, 'Adventure': 0.36}
    # user_behavior_analysis("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_15.json")
    # 10, 39, {'Romance': 0.18, 'Thriller': 0.46, 'War': 0.18, 'Action': 0.49, 'Adventure': 0.33}
    # user_behavior_analysis("../generated_user_profile/behavior_length_6/debate/user_28_profile_0_20.json")
    # 10,41,{'Romance': 0.2, 'Thriller': 0.44, 'War': 0.2, 'Action': 0.49, 'Adventure': 0.32}

    user_behavior_analysis("../generated_user_profile/task2/l3/user_28_profile.json")
    # 1, 24, {'Action': 0.54, 'Adventure': 0.33, 'Romance': 0.25, 'Thriller': 0.29, 'War': 0.33}
    # 2, 35, {'War': 0.29, 'Romance': 0.26, 'Thriller': 0.34, 'Adventure': 0.26, 'Action': 0.4, 'Film-Noir': 0.03}
    # 3, 20, {'Action': 0.45, 'Adventure': 0.45, 'Romance': 0.2, 'War': 0.2, 'Thriller': 0.35}
    # 4, 26, {'War': 0.42, 'Action': 0.38, 'Adventure': 0.12, 'Thriller': 0.38, 'Romance': 0.15}
    # 5, 12, {'War': 0.17, 'Romance': 0.25, 'Thriller': 0.33, 'Action': 0.25, 'Adventure': 0.33}
    # 6, 15, {'Romance': 0.13, 'Thriller': 0.67, 'War': 0.13, 'Action': 0.53, 'Adventure': 0.33}
    # 7, 10, {'Action': 0.4, 'Adventure': 0.3, 'War': 0.3, 'Romance': 0.2, 'Thriller': 0.4}
    # 8, 13, {'Action': 0.23, 'Thriller': 0.31, 'War': 0.23, 'Adventure': 0.46, 'Romance': 0.23}
    # 9, 7, {'War': 0.29, 'Thriller': 0.57, 'Romance': 0.43}
    # 10, 22, {'Romance': 0.18, 'Thriller': 0.41, 'War': 0.27, 'Action': 0.55, 'Adventure': 0.5}
    # 11, 6, {'Romance': 0.67, 'War': 0.33, 'Action': 0.33, 'Adventure': 0.5, 'Thriller': 0.17}
    # 12, 24, {'Action': 0.46, 'Adventure': 0.38, 'War': 0.29, 'Romance': 0.21, 'Thriller': 0.38}
    # 13, 12, {'Romance': 0.58, 'War': 0.17, 'Thriller': 0.33, 'Adventure': 0.33}
    # 14, 15, {'Romance': 0.07, 'Thriller': 0.4, 'Action': 0.47, 'Adventure': 0.67, 'War': 0.07}
    # 15, 12, {'Action': 0.5, 'Adventure': 0.5, 'Romance': 0.42, 'War': 0.17, 'Thriller': 0.33}
    # 16, 26, {'Adventure': 0.35, 'War': 0.23, 'Action': 0.38, 'Romance': 0.23, 'Thriller': 0.54, 'Film-Noir': 0.04}
    # 17, 11, {'Action': 0.55, 'Adventure': 0.36, 'Romance': 0.27, 'War': 0.27, 'Thriller': 0.55}
    # 18, 7, {'Action': 0.14, 'Thriller': 0.57, 'War': 0.14, 'Romance': 0.29, 'Adventure': 0.29}
    # 19, 6, {'War': 0.5, 'Romance': 0.33, 'Thriller': 0.67, 'Action': 0.17}
    # 20, 5, {'War': 0.4, 'Adventure': 0.4, 'Romance': 0.2, 'Thriller': 0.4, 'Action': 0.2}
    # 21, 3, {'Romance': 0.67, 'Thriller': 0.67, 'War': 0.33}
    # 22, 6, {'Adventure': 0.5, 'Romance': 0.33, 'Thriller': 0.17, 'Action': 0.17}
    # 23, 15, {'Film-Noir': 0.13, 'Thriller': 0.4, 'War': 0.13, 'Romance': 0.2, 'Adventure': 0.4, 'Action': 0.27}
    # 24, 6, {'Action': 0.17, 'Romance': 0.67, 'Thriller': 0.33, 'War': 0.17}
    # 25, 10, {'Action': 0.3, 'Adventure': 0.4, 'Romance': 0.3, 'Thriller': 0.5, 'War': 0.1}
    # 26, 11, {'Romance': 0.09, 'Thriller': 0.73, 'Action': 0.55, 'Adventure': 0.27, 'War': 0.18}
    # 27, 7, {'Thriller': 0.57, 'War': 0.29, 'Romance': 0.43}
    # 28, 5, {'Action': 0.2, 'Adventure': 0.2, 'War': 0.4, 'Romance': 0.6, 'Thriller': 0.4}
    user_behavior_analysis("../generated_user_profile/task2/l3/user_28_profile_0_19.json")
    # 1, 42, {'Action': 0.71, 'Adventure': 0.29, 'Romance': 0.14, 'Thriller': 0.24, 'War': 0.24}
    # 2, 55, {'War': 0.22, 'Romance': 0.16, 'Thriller': 0.31, 'Adventure': 0.25, 'Action': 0.58, 'Film-Noir': 0.02}
    # 3, 40, {'Action': 0.7, 'Adventure': 0.47, 'Romance': 0.1, 'War': 0.12, 'Thriller': 0.33}
    # 4, 39, {'War': 0.31, 'Action': 0.51, 'Adventure': 0.15, 'Thriller': 0.26, 'Romance': 0.15}
    # 5, 30, {'War': 0.1, 'Romance': 0.17, 'Thriller': 0.47, 'Action': 0.6, 'Adventure': 0.4}
    # 6, 36, {'Romance': 0.06, 'Thriller': 0.47, 'War': 0.08, 'Action': 0.78, 'Adventure': 0.28}
    # 7, 29, {'Action': 0.79, 'Adventure': 0.31, 'War': 0.14, 'Romance': 0.1, 'Thriller': 0.41}
    # 8, 31, {'Action': 0.65, 'Thriller': 0.42, 'War': 0.13, 'Adventure': 0.48, 'Romance': 0.16}
    # 9, 22, {'War': 0.14, 'Thriller': 0.45, 'Romance': 0.18, 'Action': 0.68, 'Adventure': 0.36}
    # 10, 42, {'Romance': 0.1, 'Thriller': 0.38, 'War': 0.21, 'Action': 0.74, 'Adventure': 0.45}
    # 11, 27, {'Romance': 0.15, 'War': 0.19, 'Action': 0.85, 'Adventure': 0.56, 'Thriller': 0.37}
    # 12, 40, {'Action': 0.6, 'Adventure': 0.28, 'War': 0.25, 'Romance': 0.12, 'Thriller': 0.38}
    # 13, 30, {'Romance': 0.3, 'War': 0.17, 'Thriller': 0.37, 'Adventure': 0.43, 'Action': 0.6}
    # 14, 37, {'Romance': 0.05, 'Thriller': 0.46, 'Action': 0.76, 'Adventure': 0.46, 'War': 0.08}
    # 15, 28, {'Action': 0.75, 'Adventure': 0.43, 'Romance': 0.25, 'War': 0.14, 'Thriller': 0.43}
    # 16, 42, {'Adventure': 0.33, 'War': 0.19, 'Action': 0.57, 'Romance': 0.17, 'Thriller': 0.36, 'Film-Noir': 0.02}
    # 17, 29, {'Action': 0.83, 'Adventure': 0.31, 'Romance': 0.1, 'War': 0.21, 'Thriller': 0.45}
    # 18, 24, {'Action': 0.67, 'Thriller': 0.42, 'War': 0.08, 'Romance': 0.17, 'Adventure': 0.42}
    # 19, 22, {'War': 0.36, 'Romance': 0.09, 'Thriller': 0.27, 'Action': 0.73, 'Adventure': 0.14}
    # 20, 23, {'War': 0.13, 'Adventure': 0.39, 'Romance': 0.04, 'Thriller': 0.43, 'Action': 0.7}
    # 21, 17, {'Romance': 0.18, 'Thriller': 0.41, 'War': 0.06, 'Action': 0.82, 'Adventure': 0.47}
    # 22, 28, {'Adventure': 0.46, 'Romance': 0.14, 'Thriller': 0.36, 'Action': 0.75, 'War': 0.11}
    # 23, 32, {'Film-Noir': 0.06, 'Thriller': 0.38, 'War': 0.12, 'Romance': 0.16, 'Adventure': 0.41, 'Action': 0.62}
    # 24, 22, {'Action': 0.68, 'Romance': 0.27, 'Thriller': 0.36, 'War': 0.14, 'Adventure': 0.18}
    # 25, 27, {'Action': 0.7, 'Adventure': 0.33, 'Romance': 0.15, 'Thriller': 0.26, 'War': 0.11}
    # 26, 29, {'Romance': 0.03, 'Thriller': 0.45, 'Action': 0.79, 'Adventure': 0.31, 'War': 0.14}
    # 27, 23, {'Thriller': 0.48, 'War': 0.09, 'Romance': 0.17, 'Action': 0.61, 'Adventure': 0.26}
    # 28, 25, {'Action': 0.76, 'Adventure': 0.48, 'War': 0.2, 'Romance': 0.16, 'Thriller': 0.32}






