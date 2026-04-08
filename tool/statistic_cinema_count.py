import json
from ftplib import print_line

from exceptiongroup import catch

from movie_len_loader import MovieLenLoader
import pandas as pd

class MovieStatistic:

    ratings,movies,users = MovieLenLoader.load_movielens_data()

    rating_df = pd.DataFrame(ratings)
    movies_df = pd.DataFrame(movies)
    users_df = pd.DataFrame(users)

    @staticmethod
    def get_all_behavior(profile_path):
        user_profile_list = []
        with open(profile_path, 'r', encoding='utf-8') as file:
            user_profile_list = json.load(file)
        movie_id_and_info_map = MovieLenLoader.get_movie_id_and_info_map()
        movie_set = set()
        movie_genre_and_count_map = {}
        for user_profile in user_profile_list:
            single_user_bahavior = user_profile['behavior']
            for behavior in single_user_bahavior:
                behavior = str(behavior)
                if behavior in movie_set:
                    # print("movieId:{} already exists".format(behavior))
                    continue
                movie_detail = movie_id_and_info_map.get(behavior)
                movie_set.add(behavior)
                movie_genres = ''
                try:
                    movie_genres = movie_detail['genres']
                except:
                    print("cannot load genres,movie_id:",behavior)
                # 拆分genres
                movie_genre_list = movie_genres.split('|')
                # 统计个数
                for genre in movie_genre_list:
                    if movie_genre_and_count_map.get(genre) is not None:
                        movie_genre_and_count_map[genre] += 1
                    else:
                        movie_genre_and_count_map[genre] = 1

        return movie_genre_and_count_map,movie_set



if __name__ == '__main__':
    movie_genre_and_count_map,movie_set = MovieStatistic().get_all_behavior()
    print(movie_genre_and_count_map)
    for genre in movie_genre_and_count_map:
        rate = movie_genre_and_count_map[genre]/len(movie_set)
        print_line(f"genre:{genre} rate:{rate}")
    print(movie_set.__len__())