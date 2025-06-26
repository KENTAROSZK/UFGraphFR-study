"""
Copyright 2025 trueWangSyutung
Open Academic Community License V1
"""

import copy
import logging

import numpy as np
import torch
from sklearn.metrics import pairwise_distances


# Checkpoints
def save_checkpoint(model, model_dir):
    torch.save(model.state_dict(), model_dir)


def resume_checkpoint(model, model_dir, device_id):
    state_dict = torch.load(
        model_dir, map_location=lambda storage, loc: storage.cuda(device=device_id)
    )  # ensure all storage are on gpu
    model.load_state_dict(state_dict)


# Hyper params
def use_cuda(enabled, device_id=0):
    if enabled:
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.set_device(device_id)


def use_mps(enabled):
    if enabled and torch.backends.mps.is_available():
        torch.device("mps")
        return torch.device("mps")


def use_optimizer(network, params):
    if params["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(
            network.parameters(),
            lr=params["sgd_lr"],
            momentum=params["sgd_momentum"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "adam":
        optimizer = torch.optim.Adam(
            network.parameters(),
            lr=params["lr"],
            weight_decay=params["l2_regularization"],
        )
    elif params["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(
            network.parameters(),
            lr=params["rmsprop_lr"],
            alpha=params["rmsprop_alpha"],
            momentum=params["rmsprop_momentum"],
        )
    return optimizer


def construct_user_relation_graph_via_item(
    round_user_params, item_num, latent_dim, similarity_metric
):
    # prepare the item embedding array.
    item_embedding = np.zeros(
        (len(round_user_params), item_num * latent_dim), dtype="float32"
    )
    for user in round_user_params.keys():
        item_embedding[user] = (
            round_user_params[user]["embedding_item.weight"].numpy().flatten()
        )
    # construct the user relation graph.
    adj = pairwise_distances(item_embedding, metric=similarity_metric)
    if similarity_metric == "cosine":
        return adj
    else:
        return -adj


def construct_user_relation_graph_via_random(
    round_user_params, latent_dim, similarity_metric
):
    adj = np.zeros((len(round_user_params), len(round_user_params)), dtype="float32")
    # construct the user relation graph by randomly sampling.
    # 随机选取 user a 和 user b 作为邻居，并计算相似度。
    for user in range(adj.shape[0]):
        for neighbor in range(adj.shape[1]):
            if user != neighbor:
                adj[user][neighbor] = np.random.randint(-5, 6) / 10
    return adj


def construct_user_relation_graph_via_user(
    round_user_params, latent_dim, similarity_metric
):
    # prepare the user embedding array.
    # Get the actual shape from the first user's embedding
    first_user_key = next(iter(round_user_params.keys()))
    actual_embedding_weights = round_user_params[first_user_key][
        "embedding_user.weight"
    ].numpy()
    # For Linear layer, the shape is (in_features, out_features), we need to flatten it
    embedding_size = actual_embedding_weights.size  # Total number of elements
    user_embedding = np.zeros((len(round_user_params), embedding_size), dtype="float32")
    for user in round_user_params.keys():
        user_embedding[user] = copy.deepcopy(
            round_user_params[user]["embedding_user.weight"].numpy().flatten()
        )
    # construct the user relation graph.
    adj = pairwise_distances(user_embedding, metric=similarity_metric)
    if similarity_metric == "cosine":
        return adj
    else:
        return -adj


def select_topk_neighboehood(
    user_realtion_graph, neighborhood_size, neighborhood_threshold
):
    topk_user_relation_graph = np.zeros(user_realtion_graph.shape, dtype="float32")
    if neighborhood_size > 0:
        for user in range(user_realtion_graph.shape[0]):
            user_neighborhood = user_realtion_graph[user]
            topk_indexes = user_neighborhood.argsort()[-neighborhood_size:][::-1]
            for i in topk_indexes:
                topk_user_relation_graph[user][i] = 1 / neighborhood_size
    else:
        similarity_threshold = np.mean(user_realtion_graph) * neighborhood_threshold
        for i in range(user_realtion_graph.shape[0]):
            high_num = np.sum(user_realtion_graph[i] > similarity_threshold)
            if high_num > 0:
                for j in range(user_realtion_graph.shape[1]):
                    if user_realtion_graph[i][j] > similarity_threshold:
                        topk_user_relation_graph[i][j] = 1 / high_num
            else:
                topk_user_relation_graph[i][i] = 1

    return topk_user_relation_graph


def MP_on_graph_with_embedding_user(
    round_user_params, item_num, latent_dim, topk_user_relation_graph, layers
):
    # prepare the item embedding array.
    user_embedding = np.zeros(
        (len(round_user_params), 100 * latent_dim), dtype="float32"
    )
    for user in round_user_params.keys():
        user_embedding[user] = (
            round_user_params[user]["embedding_user.weight"].numpy().flatten()
        )

    # aggregate item embedding via message passing.
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, user_embedding)
    for layer in range(layers - 1):
        aggregated_item_embedding = np.matmul(
            topk_user_relation_graph, aggregated_item_embedding
        )

    # reconstruct item embedding.
    user_embedding_dict = {}
    for user in round_user_params.keys():
        user_embedding_dict[user] = torch.from_numpy(
            aggregated_item_embedding[user].reshape(latent_dim, 100)
        )
    user_embedding_dict["global"] = sum(user_embedding_dict.values()) / len(
        round_user_params
    )
    return user_embedding_dict


def MP_on_graph(
    round_user_params, item_num, latent_dim, topk_user_relation_graph, layers
):
    # prepare the item embedding array.
    item_embedding = np.zeros(
        (len(round_user_params), item_num * latent_dim), dtype="float32"
    )
    for user in round_user_params.keys():
        item_embedding[user] = (
            round_user_params[user]["embedding_item.weight"].numpy().flatten()
        )

    # aggregate item embedding via message passing.
    aggregated_item_embedding = np.matmul(topk_user_relation_graph, item_embedding)
    for layer in range(layers - 1):
        aggregated_item_embedding = np.matmul(
            topk_user_relation_graph, aggregated_item_embedding
        )

    # reconstruct item embedding.
    item_embedding_dict = {}
    for user in round_user_params.keys():
        item_embedding_dict[user] = torch.from_numpy(
            aggregated_item_embedding[user].reshape(item_num, latent_dim)
        )
    item_embedding_dict["global"] = sum(item_embedding_dict.values()) / len(
        round_user_params
    )
    return item_embedding_dict


def initLogging(logFilename):
    """Init for logging"""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s-%(levelname)s-%(message)s",
        datefmt="%y-%m-%d %H:%M",
        filename=logFilename,
        filemode="w",
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    console.setFormatter(formatter)
    logging.getLogger("").addHandler(console)


def compute_regularization(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction="mean")
    for name, param in model.named_parameters():
        if name == "embedding_item.weight":
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss


def compute_regularization2(model, parameter_label):
    reg_fn = torch.nn.MSELoss(reduction="mean")
    for name, param in model.named_parameters():
        if name == "embedding_user.weight":
            reg_loss = reg_fn(param, parameter_label)
            return reg_loss
