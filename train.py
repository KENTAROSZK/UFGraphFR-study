"""
Copyright 2025 trueWangSyutung
Open Academic Community License V1
"""

import warnings

from embedding import EmbeddingUtils

warnings.filterwarnings("ignore")
import datetime
import json
import os

import numpy as np
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
import argparse

from data import SampleGenerator
from mlp import MLPEngine
from mymodel import UFGraphFREngine as UFGraphFREngine
from utils import *

# Training settings UFGraphFR
parser = argparse.ArgumentParser()
parser.add_argument("--alias", type=str, default="UFGraphFR")
parser.add_argument("--clients_sample_ratio", type=float, default=1.0)
parser.add_argument("--clients_sample_num", type=int, default=0)
parser.add_argument("--num_round", type=int, default=100)
parser.add_argument("--local_epoch", type=int, default=1)
parser.add_argument("--construct_graph_source", type=str, default="item")
parser.add_argument("--neighborhood_size", type=int, default=0)
parser.add_argument("--neighborhood_threshold", type=float, default=1.0)
parser.add_argument("--mp_layers", type=int, default=1)
parser.add_argument("--similarity_metric", type=str, default="cosine")
parser.add_argument("--reg", type=float, default=1.0)
parser.add_argument("--lr_eta", type=int, default=80)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--optimizer", type=str, default="sgd")
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--dataset", type=str, default="100k")
parser.add_argument("--num_users", type=int)
parser.add_argument("--num_items", type=int)
parser.add_argument("--latent_dim", type=int, default=32)
parser.add_argument("--num_negative", type=int, default=4)
parser.add_argument("--layers", type=str, default="64, 32, 16, 8")
parser.add_argument("--l2_regularization", type=float, default=0.0)
parser.add_argument("--dp", type=float, default=0.1)
parser.add_argument("--use_cuda", type=bool, default=False)
parser.add_argument("--use_mps", type=bool, default=False)
parser.add_argument("--use_transfermer", type=bool, default=True)
parser.add_argument("--use_jointembedding", type=bool, default=True)

parser.add_argument("--device_id", type=int, default=0)
parser.add_argument(
    "--model_dir", type=str, default="checkpoints/{}_Epoch{}_HR{:.4f}_NDCG{:.4f}.model"
)
parser.add_argument("--ind", type=int, default=0)
parser.add_argument("--ps", type=str, default=None)
# parser.add_argument('--grid_size', type=str, default='3, 3, 3, 3')
# update_round
parser.add_argument("--update_round", type=int, default=1)
parser.add_argument("--embed_dim", type=int, default=100)
parser.add_argument("--pre_model", type=str, default="USE")

args = parser.parse_args()
config = vars(args)


def train(config):
    if type(config["layers"]) is str:
        # Model.
        if len(config["layers"]) > 1:
            config["layers"] = [int(item) for item in config["layers"].split(",")]
        else:
            config["layers"] = int(config["layers"])

    if config["dataset"] == "ml-1m":
        config["num_users"] = 6040  # 用户数
        config["num_items"] = 3706  # 物品数
    elif config["dataset"] == "100k":
        config["num_users"] = 943  # 用户数
        config["num_items"] = 1682  # 物品数
    elif config["dataset"] == "lastfm-2k":
        config["num_users"] = 1600  # 用户数 1600 1484
        config["num_items"] = 12545  # 物品数
    elif config["dataset"] == "amazon":
        config["num_users"] = 8072  # 用户数
        config["num_items"] = 11830  # 物品数
    elif config["dataset"] == "hetres-2k":
        config["num_users"] = 2113  # 用户数
        config["num_items"] = 10109  # 物品数
    elif config["dataset"] == "kuai-small":
        config["num_users"] = 1411
        config["num_items"] = 1932
    elif config["dataset"] == "douban":
        config["num_users"] = 1905
        config["num_items"] = 24161

    else:
        pass

    if (
        config["alias"] == "UFGraphFR"
        or config["alias"] == "UFGraphFR-lite"
        or config["alias"] == "UFGraphFR-pre"
    ):
        engine = UFGraphFREngine(config)
    else:
        engine = MLPEngine(config)

    user_infos = None
    # Load Data
    dataset_dir = "data/" + config["dataset"] + "/" + "ratings.dat"
    if config["dataset"] == "ml-1m":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        # UserID::Gender::Age::Occupation::Zip-code

        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "u.user",
            sep=",",
            header=None,
            names=["uid", "age", "gender", "occupation", "zipcode"],
            engine="python",
        )

    elif config["dataset"] == "100k":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "u.user",
            sep=",",
            header=None,
            names=["uid", "age", "gender", "occupation", "zipcode"],
            engine="python",
        )
    elif config["dataset"] == "lastfm-2k":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "user.dat",
            sep=",",
            header=None,
            names=["uid", "tag"],
            engine="python",
        )
    elif config["dataset"] == "hetres-2k":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "user.dat",
            sep=",",
            header=None,
            names=["uid", "tag"],
            engine="python",
        )
    elif config["dataset"] == "kuai-small":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        # user_id,user_active_degree,is_lowactive_period,is_live_streamer,is_video_author,follow_user_num,
        # follow_user_num_range,fans_user_num,fans_user_num_range,friend_user_num,friend_user_num_range,
        # register_days,register_days_range,onehot_feat0,onehot_feat1,onehot_feat2,onehot_feat3,onehot_feat4,
        # onehot_feat5,onehot_feat6,onehot_feat7,onehot_feat8,onehot_feat9,onehot_feat10,onehot_feat11,onehot_feat12,
        # onehot_feat13,onehot_feat14,onehot_feat15,onehot_feat16,onehot_feat17
        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "u.user",
            sep=",",
            header=None,
            names=[
                "uid",
                "user_active_degree",
                "is_lowactive_period",
                "is_live_streamer",
                "is_video_author",
                "follow_user_num",
                "follow_user_num_range",
                "fans_user_num",
                "fans_user_num_range",
                "friend_user_num",
                "friend_user_num_range",
                "register_days",
                "register_days_range",
                "onehot_feat0",
                "onehot_feat1",
                "onehot_feat2",
                "onehot_feat3",
                "onehot_feat4",
                "onehot_feat5",
                "onehot_feat6",
                "onehot_feat7",
                "onehot_feat8",
                "onehot_feat9",
                "onehot_feat10",
                "onehot_feat11",
                "onehot_feat12",
                "onehot_feat13",
                "onehot_feat14",
                "onehot_feat15",
                "onehot_feat16",
                "onehot_feat17",
            ],
            engine="python",
        )

    elif config["dataset"] == "douban":
        # user_id,item_id,rating,time,item_type

        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp", "item_type", "count"],
            engine="python",
        )
        user_infos = pd.read_csv(
            "data/" + config["dataset"] + "/" + "user.dat",
            sep=",",
            names=["uid", "living_place", "join_time", "self_statement"],
            engine="python",
        )

    elif config["dataset"] == "amazon":
        rating = pd.read_csv(
            dataset_dir,
            sep=",",
            header=None,
            names=["uid", "mid", "rating", "timestamp"],
            engine="python",
        )
        rating = rating.sort_values(by="uid", ascending=True)

    else:
        pass
    # Reindex
    user_id = rating[["uid"]].drop_duplicates().reindex()
    user_id["userId"] = np.arange(len(user_id))
    rating = pd.merge(rating, user_id, on=["uid"], how="left")
    item_id = rating[["mid"]].drop_duplicates()
    item_id["itemId"] = np.arange(len(item_id))
    rating = pd.merge(rating, item_id, on=["mid"], how="left")
    rating = rating[["userId", "itemId", "rating", "timestamp"]]
    print(
        "Range of userId is [{}, {}]".format(rating.userId.min(), rating.userId.max())
    )
    print(
        "Range of itemId is [{}, {}]".format(rating.itemId.min(), rating.itemId.max())
    )
    print("-" * 80)
    print("Data Loading Done !")

    # DataLoader for training
    sample_generator = SampleGenerator(ratings=rating)
    validate_data = sample_generator.validate_data
    test_data = sample_generator.test_data
    # np.save('model_parameter/' + str(config['ind']) + config['dataset'] + '-' + 'test_data-2.npy', test_data)
    embeddingUtils = EmbeddingUtils(
        user_infos=user_infos, config=config, dataset=config["dataset"]
    )

    hit_ratio_list = []
    ndcg_list = []
    val_hr_list = []
    val_ndcg_list = []
    train_loss_list = []
    test_loss_list = []
    val_loss_list = []
    best_val_hr = 0
    final_test_round = 0
    for round in range(config["num_round"]):
        # break
        print("-" * 80)
        print("Round {} starts !".format(round))

        all_train_data = sample_generator.store_all_train_data(config["num_negative"])
        print("-" * 80)
        print("Training phase!")
        tr_loss = engine.fed_train_a_round(
            all_train_data, round_id=round, embeddingUtils=embeddingUtils
        )
        # break
        train_loss_list.append(tr_loss)

        print("-" * 80)
        print("Testing phase!")
        hit_ratio, ndcg, te_loss = engine.fed_evaluate(test_data, embeddingUtils)
        test_loss_list.append(te_loss)
        # break
        print(
            "[Testing Epoch {}] HR = {:.4f}, NDCG = {:.4f}".format(
                round, hit_ratio, ndcg
            )
        )
        hit_ratio_list.append(hit_ratio)
        ndcg_list.append(ndcg)

        print("-" * 80)
        print("Validating phase!")
        val_hit_ratio, val_ndcg, v_loss = engine.fed_evaluate(
            validate_data, embeddingUtils
        )
        val_loss_list.append(v_loss)
        print(
            "[Evluating Epoch {}] HR = {:.4f}, NDCG = {:.4f}".format(
                round, val_hit_ratio, val_ndcg
            )
        )
        val_hr_list.append(val_hit_ratio)
        val_ndcg_list.append(val_ndcg)

        if val_hit_ratio >= best_val_hr:
            best_val_hr = val_hit_ratio
            final_test_round = round
            # np.save('model_parameter/' + str(config['ind']) + config['dataset'] + '-' + 'client_param-2.npy', engine.client_model_params)

    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
    result_str = (
        current_time
        + "-"
        + "layers: "
        + str(config["layers"])
        + "-"
        + "lr: "
        + str(config["lr"])
        + "-"
        + "clients_sample_ratio: "
        + str(config["clients_sample_ratio"])
        + "-"
        + "num_round: "
        + str(config["num_round"])
        + "-"
        "neighborhood_size: "
        + str(config["neighborhood_size"])
        + "-"
        + "mp_layers: "
        + str(config["mp_layers"])
        + "-"
        + "negatives: "
        + str(config["num_negative"])
        + "-"
        + "lr_eta: "
        + str(config["lr_eta"])
        + "-"
        + "batch_size: "
        + str(config["batch_size"])
        + "-"
        + "hr: "
        + str(hit_ratio_list[final_test_round])
        + "-"
        + "ndcg: "
        + str(ndcg_list[final_test_round])
        + "-"
        + "best_round: "
        + str(final_test_round)
        + "-"
        + "similarity_metric: "
        + str(config["similarity_metric"])
        + "-"
        + "neighborhood_threshold: "
        + str(config["neighborhood_threshold"])
        + "-"
        + "reg: "
        + str(config["reg"])
        + "-"
        + "ps: "
        + str(config["ps"])
        + "-"
        + "construct_graph_source: "
        + str(config["construct_graph_source"])
        + "-"
        + "dataset: "
        + str(config["dataset"])
    )
    file_name = (
        "sh_result/"
        + config["construct_graph_source"]
        + "-"
        + config["dataset"]
        + ".txt"
    )
    with open(file_name, "a") as file:
        file.write(result_str + "\n")

    print(config["alias"])
    print(
        "clients_sample_ratio: {}, lr_eta: {}, bz: {}, lr: {}, dataset: {}, layers: {}, negatives: {}, "
        "neighborhood_size: {}, neighborhood_threshold: {}, mp_layers: {}, similarity_metric: {}, "
        "construct_graph_source : {}".format(
            config["clients_sample_ratio"],
            config["lr_eta"],
            config["batch_size"],
            config["lr"],
            config["dataset"],
            config["layers"],
            config["num_negative"],
            config["neighborhood_size"],
            config["neighborhood_threshold"],
            config["mp_layers"],
            config["similarity_metric"],
            config["construct_graph_source"],
        )
    )

    print("hit_list: {}".format(hit_ratio_list))
    print("ndcg_list: {}".format(ndcg_list))
    print("val_hr_list: {}".format(val_hr_list))

    print("val_ndcg_list: {}".format(val_ndcg_list))
    with open(
        "res/"
        + "model-"
        + str(config["alias"])
        + "dataset-"
        + str(config["dataset"])
        + "latent_dim-"
        + str(config["latent_dim"])
        + "use_jointembedding-"
        + str(config["use_jointembedding"])
        + "use_transfermer-"
        + str(config["use_transfermer"])
        + "reg-"
        + str(config["reg"])
        + ".jsonl",
        "a",
    ) as log_file:
        jsonobj = {
            "alias": config["alias"],
            "layers": config["layers"],
            "latent_dim": config["latent_dim"],
            "dataset": config["dataset"],
            "reg": config["reg"],
            "hit_list": hit_ratio_list,
            "ndcg_list": ndcg_list,
            "val_hr_list": val_hr_list,
            "val_ndcg_list": val_ndcg_list,
            "train_loss_list": train_loss_list,
            "test_loss_list": test_loss_list,
            "val_loss_list": val_loss_list,
            "best_val_hr": best_val_hr,
            "final_test_round": final_test_round,
            "result_str": result_str,
        }
        # 写入 json 数据到文件中
        log_file.write("\n")

        log_file.write(json.dumps(jsonobj, ensure_ascii=False))

    # 关闭日志文件

    return "Done!"


if __name__ == "__main__":
    # 从 0.1～1.0 ， 0.1 的步长， 10 个点
    train(config)
