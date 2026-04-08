import json
import os


class RecHistoryProcessor:

    @staticmethod
    def clean(rec_history_path = '../data/rec_history.json'):
        if os.path.exists(rec_history_path):
            os.remove(rec_history_path)

    @staticmethod
    def get_rec_history_list(rec_history_path = '../data/rec_history.json'):
        with open(rec_history_path, 'r', encoding='utf-8') as f:
            rec_history_list = json.load(f)
        return rec_history_list

    @staticmethod
    def get_average_count(rec_history_path = '../data/rec_history.json'):
        rec_history_list = RecHistoryProcessor.get_rec_history_list(rec_history_path)
        sum = 0
        for index in range(len(rec_history_list)):
            sum += len(rec_history_list[index])
        return sum/len(rec_history_list)

    @staticmethod
    def load_rec_history(file_path):
        rec_history_path = '../data/rec_history.json'
        rec_history_list = RecHistoryProcessor.get_rec_history_list(file_path)
        json.dump(rec_history_list, open(rec_history_path, 'w', encoding='utf-8'), indent=4)

