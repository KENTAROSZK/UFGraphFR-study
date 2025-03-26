'''
Copyright 2025 trueWangSyutung
Open Academic Community License V1 
'''
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import text
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

class EmbeddingUtils():
    def embedding_uder_info(self,user_info):
        if self.config['pre_model'] == "USE":
            user_emb = self.embedder.embed(user_info)[0]
            return user_emb
        pass
    def embedding_dataset(self,user_infos):
        if self.config['pre_model'] == "USE":
            # 创建一个 config['item_num'] * config['item_dim'] 的矩阵，每个元素都是 0
            user_emb = np.zeros((len(user_infos), self.config['embed_dim']))
            # item_emb = np.zeros((len(item_infos), self.config['embed_dim']))
            for i in range(len(user_infos)):
                user_emb[i] = self.embedder.embed(user_infos[i])[0]

            #for i in range(len(item_infos)):
            #    item_emb[i] = self.embedder.embed(item_infos[i])[0]


            return  user_emb

    def __init__(self,config,user_infos,dataset):
        self.config = config

        if config['pre_model'] == "USE":
            model_path = "universal_sentence_encoder.tflite"
            BaseOptions = mp.tasks.BaseOptions
            TextEmbedder = mp.tasks.text.TextEmbedder
            TextEmbedderOptions = mp.tasks.text.TextEmbedderOptions
            # For creating a text embedder instance: /Users/openacademic/Downloads/douban_dataset(text information)/UFGraphFR/universal_sentence_encoder.tflite
            options = TextEmbedderOptions(
                base_options=BaseOptions(model_asset_path=model_path),
                quantize=True)
            self.embedder = TextEmbedder.create_from_options(options)
        elif config['pre_model'] == "MiniLM-L6":
            self.embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.user_infos = user_infos
        # 将 self.user_infos 中 的 NaN 元素替换成 “ ”
        self.user_infos = self.user_infos.fillna("")
        
        print("EmbeddingUtils init",self.user_infos)
        # self.item_infos = item_infos
        self.dataset = config['dataset']


    def embedding_users(self,user_id):
        user = None
        prompts = ""
        if self.dataset == "ml-1m":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            prompts = "The user'id is {} and his gender is {}, he is {} years old, works in the field of {} and lives at zip code {}".format(user["uid"], user["gender"], user["age"], user["occupation"], user["zipcode"])
        elif self.dataset == "100k":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            prompts = "The user'id is {} and his gender is {}, he is {} years old, works in the field of {} and lives at zip code {}".format(user["uid"], user["gender"], user["age"], user["occupation"], user["zipcode"])
           
        elif self.dataset == "kuai-small":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            # user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,follow_user_num,
            # follow_user_num_range,fans_user_num,fans_user_num_range,friend_user_num,friend_user_num_range,
            # register_days,register_days_range,onehot_feat0,onehot_feat1,onehot_feat2,onehot_feat3,onehot_feat4,
            # onehot_feat5,onehot_feat6,onehot_feat7,onehot_feat8,onehot_feat9,onehot_feat10,onehot_feat11,onehot_feat12,
            # onehot_feat13,onehot_feat14,onehot_feat15,onehot_feat16,onehot_feat17
            # user_infos = pd.read_csv("data/" + config['dataset'] + "/" + "user_info.txt", sep=",", header=None, 
            #                     names=['uid', 'user_active_degree', 'is_lowactive_period', 
            #                            'is_live_streamer', 'is_video_author', 'follow_user_num', 
            #                            'follow_user_num_range', 'fans_user_num', 'fans_user_num_range', 
            #                            'friend_user_num', 'friend_user_num_range', 'register_days',
            #                              'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
            #                                'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
            #                                'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 
            #                                'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
            #                                'onehot_feat15', 'onehot_feat16', 'onehot_feat17'], engine='python')
            # 转为 json
            sx = ['uid', 'user_active_degree', 'is_lowactive_period', 
                                        'is_live_streamer', 'is_video_author', 'follow_user_num', 
                                        'follow_user_num_range', 'fans_user_num', 'fans_user_num_range', 
                                        'friend_user_num', 'friend_user_num_range', 'register_days',
                                          'register_days_range', 'onehot_feat0', 'onehot_feat1', 'onehot_feat2',
                                            'onehot_feat3', 'onehot_feat4', 'onehot_feat5', 'onehot_feat6',
                                            'onehot_feat7', 'onehot_feat8', 'onehot_feat9', 'onehot_feat10', 
                                            'onehot_feat11', 'onehot_feat12', 'onehot_feat13', 'onehot_feat14',
                                            'onehot_feat15', 'onehot_feat16', 'onehot_feat17']
            prompts = ""
            mb = "The user's {} is {}. "
            for i in range(len(sx)):
                prompts += mb.format(sx[i], user[sx[i]])       
        elif self.dataset == "douban":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            # 'uid', 'living_place', 'join_time', 'self_statement'
            prompts = "The user'id is {} and his is from {}, he join the douban at {}, his self-introduction is {}.".format(user["uid"], user["living_place"], user["join_time"], user["self_statement"])
            
        elif self.dataset == "lastfm-2k":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            # user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,follow_user_num,
            # follow_user_num_range,fans_user_num,fans_user_num_range,friend_user_num,friend_user_num_range,
            # register_
            prompts = "The user'id is {} and his look {} items".format(user["uid"], user["tag"])
            
        elif self.dataset == "hetres-2k":
            user = self.user_infos[self.user_infos['uid'] == user_id]
            # user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,follow_user_num,
            # follow_user_num_range,fans_user_num,fans_user_num_range,friend_user_num,friend_user_num_range,
            # register_
            prompts = "The user'id is {} and his look {} items".format(user["uid"], user["tag"])
        
        if self.config['pre_model'] == "USE":
            embeds = self.embedder.embed(prompts)
            return torch.tensor(embeds.embeddings[0].embedding)
        elif self.config['pre_model'] == "MiniLM-L6":
            embeds = self.embedder.encode([prompts])
            return torch.tensor(embeds[0])

    



if __name__ == "__main__":
    infinity_api_url = "http://api.wlai.vip/v1"
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    promptes="The user'id is 1 and his gender is Male, he is 20 years old, works in the field of 1 and lives at zip code 1"
    r =   embedder.encode([promptes])
    print(r[0].shape)
        