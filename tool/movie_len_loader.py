# 安装必要库
# pip install pandas numpy scikit-learn matplotlib seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sympy import false

# 设置中文字体和图形样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

class MovieLenLoader(object):
    data_path = '..\\data\\ml-1m'
    # 读取评分数据（.dat格式使用::分隔）
    ratings = pd.read_csv(f'{data_path}\\ratings.dat',
                          sep='::',
                          engine='python',
                          names=['userId', 'movieId', 'rating', 'timestamp'])

    # 读取电影数据
    movies = pd.read_csv(f'{data_path}\\movies.dat',
                         sep='::',
                         engine='python',
                         names=['movieId', 'title', 'genres'],
                         encoding='latin-1')

    # 2. 加载用户数据
    users = pd.read_csv(
        f'{data_path}\\users.dat',
        sep='::',
        engine='python',
        names=['userId', 'gender', 'age', 'occupation', 'zipcode'],
        dtype={'userId': 'int32', 'age': 'int8', 'occupation': 'int8'}
    )

    movie_id_and_info_map = {}
    for index, row in movies.iterrows():
        movieId = str(row['movieId'])
        movie_id_and_info_map[movieId] = row

    # 筛选出评分>=4的
    like_ratings = ratings[ratings['rating']>=4]
    sorted_like_ratings = like_ratings.sort_values('timestamp', ascending=True)
    user_id_and_behavior_map = {}
    for user_id, group in sorted_like_ratings.groupby('userId'):
        user_id_and_behavior_map[user_id] = group['movieId'].tolist()

    movie_genre_and_count_map = {}
    for movie_id in movie_id_and_info_map:
        movie_info = movie_id_and_info_map[movie_id]
        movie_genres = ''
        try:
            movie_genres = movie_info['genres']
        except:
            print("cannot load genres,movie_id:", movie_id)
        # 拆分genres
        movie_genre_list = movie_genres.split('|')
        # 统计个数
        for genre in movie_genre_list:
            if movie_genre_and_count_map.get(genre) is not None:
                movie_genre_and_count_map[genre] += 1
            else:
                movie_genre_and_count_map[genre] = 1

    @staticmethod
    def load_movielens_data():
        return MovieLenLoader.ratings, MovieLenLoader.movies, MovieLenLoader.users

    @staticmethod
    def get_movie_id_and_info_map():
        return MovieLenLoader.movie_id_and_info_map

    @staticmethod
    def get_user_id_and_behavior_map():
        return MovieLenLoader.user_id_and_behavior_map


    @staticmethod
    def get_movie_genre_and_count_map():
        return MovieLenLoader.movie_genre_and_count_map

    @staticmethod
    def get_movie_count_in_target_tag_list(target_tag_list):
        result = 0
        movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
        for movie_id in movie_id_and_info_map:
            movie_info = movie_id_and_info_map[movie_id]
            movie_genres = ''
            try:
                movie_genres = movie_info['genres']
            except:
                print("cannot load genres,movie_id:", movie_id)
            # 拆分genres
            movie_genre_list = movie_genres.split('|')
            # 统计个数
            is_in_target_tag_list = false
            for genre in movie_genre_list:
                if genre in target_tag_list:
                    is_in_target_tag_list = True
                    break
            if is_in_target_tag_list:
                result = result + 1
        return result



