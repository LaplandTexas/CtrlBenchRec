import json

class AmazonLoader(object):
    # 先定义为空，防止 import 时直接读取大文件
    user_id_and_item_id_map = None
    item_id_and_info_map = None
    catagory_count_map = None

    @classmethod
    def _initialize(cls):
        """内部初始化方法，只在第一次调用时运行"""
        if cls.item_id_and_info_map is None:
            with open('../data/filtered_amazon_toys_and_games/user_id_and_item_id_map.json', 'r') as f:
                cls.user_id_and_item_id_map = json.load(f)
            with open('../data/filtered_amazon_toys_and_games/item_id_and_info_map.json', 'r') as f:
                cls.item_id_and_info_map = json.load(f)

            # 统计逻辑移动到这里
            cls.catagory_count_map = {}
            for item_id in cls.item_id_and_info_map:
                catagories = cls.item_id_and_info_map[item_id]['Category']
                for catagory in catagories:
                    if catagory not in cls.catagory_count_map:
                        cls.catagory_count_map[catagory] = 1
                    else:
                        cls.catagory_count_map[catagory] += 1
            cls.catagory_count_map = dict(sorted(cls.catagory_count_map.items(), key=lambda x: x[1], reverse=True))

    @staticmethod
    def get_user_id_and_item_id_map():
        AmazonLoader._initialize() # 确保数据已加载
        return AmazonLoader.user_id_and_item_id_map

    @staticmethod
    def get_item_id_and_info_map():
        AmazonLoader._initialize()
        return AmazonLoader.item_id_and_info_map

    @staticmethod
    def get_catagory_count_map():
        AmazonLoader._initialize()
        return AmazonLoader.catagory_count_map

    @staticmethod
    def get_count_in_target_tag_list(target_tag_list = []):
        AmazonLoader._initialize()
        # 性能优化：直接从已生成的 catagory_count_map 中取值，不需要重新遍历 item_id_and_info_map
        count = 0
        current_map = AmazonLoader.get_catagory_count_map()
        for tag in target_tag_list:
            count += current_map.get(tag, 0)
        return count