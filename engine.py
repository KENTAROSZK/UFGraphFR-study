'''
Copyright 2025 trueWangSyutung
Open Academic Community License V1 
'''
import torch
from torch.autograd import Variable
from tensorboardX import SummaryWriter
import tqdm

from utils import *
from metrics import MetronAtK
import random
import copy
from data import UserItemRatingDataset
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace


class Engine(object):
    """Meta Engine for training & evaluating NCF model

    Note: Subclass should implement self.model !
    """

    def __init__(self, config):
        self.config = config  # model configuration
        self._metron = MetronAtK(top_k=10)
        # self._writer = SummaryWriter(log_dir='runs/{}'.format(config['alias']))  # tensorboard writer
        # self._writer.add_text('config', str(config), 0)
        self.server_model_param = {}
        self.client_model_params = {}
        # explicit feedback
        # self.crit = torch.nn.MSELoss()
        # implicit feedback
        self.crit = torch.nn.BCELoss()
        self.top_k = 10
        self.user_relation_graph = None

    def instance_user_train_loader(self, user_train_data):
       

        """instance a user's train loader."""
        dataset = UserItemRatingDataset(user_tensor=torch.LongTensor(user_train_data[0]),
                                        item_tensor=torch.LongTensor(user_train_data[1]),
                                        target_tensor=torch.FloatTensor(user_train_data[2])
                                        )
        return DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)
    def fed_train_single_batch_my(self, model_client, batch_data, optimizers, user,user_embedding):
        """train a batch and return an updated model."""
        users, items, ratings= batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()

        reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data) if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot'  else copy.deepcopy(self.server_model_param['embedding_user.weight'][user].data)

        optimizer, optimizer_i, optimizer_u = optimizers
        
        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()
        if self.config['use_mps'] is True:
            device = torch.device("mps")
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            reg_item_embedding = reg_item_embedding.to(device)

        optimizer.zero_grad()
        optimizer_i.zero_grad()            
        if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-pre':
            # optimizer_t.zero_grad()
            if self.config['use_jointembedding']:
                optimizer_u.zero_grad()
            items = [items]
        ratings_pred = model_client.forward(items,user_embedding)
        loss = self.crit(ratings_pred.view(-1), ratings)
        regularization_term = compute_regularization(model_client, reg_item_embedding) if self.config['alias'] == 'UFGraphFR' else compute_regularization2(model_client, reg_item_embedding)

        loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        optimizer_i.step()
        if self.config['alias'] == 'UFGraphFR':
            if self.config['use_jointembedding']:
                optimizer_u.step()
        
        return model_client, loss.item()

    def fed_train_single_batch(self, model_client, batch_data, optimizers, user):
        """train a batch and return an updated model."""
        users, items, ratings= batch_data[0], batch_data[1], batch_data[2]
        ratings = ratings.float()
        reg_item_embedding = None
        if self.config['alias'] == 'UFGraphFR':
            reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)
        elif self.config['alias'] == 'UFGraphFR-pre':
            reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_user.weight'][user].data)
        else:
            reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)

        # reg_item_embedding = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)
        optimizer, optimizer_u, optimizer_i,optimizer_t = None,None,None,None

        if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
            optimizer, optimizer_u, optimizer_i = optimizers
        else:
            optimizer, optimizer_u, optimizer_i = optimizers

        if self.config['use_cuda'] is True:
            users, items, ratings = users.cuda(), items.cuda(), ratings.cuda()
            reg_item_embedding = reg_item_embedding.cuda()
        if self.config['use_mps'] is True:
            device = torch.device("mps")
            users, items, ratings = users.to(device), items.to(device), ratings.to(device)
            reg_item_embedding = reg_item_embedding.to(device)
        optimizer.zero_grad()
        optimizer_u.zero_grad()
        optimizer_i.zero_grad()            
        if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
            # optimizer_t.zero_grad()
            items = [items]
        ratings_pred = model_client(items)
        loss = self.crit(ratings_pred.view(-1), ratings)
        regularization_term = compute_regularization(model_client, reg_item_embedding)
        loss += self.config['reg'] * regularization_term
        loss.backward()
        optimizer.step()
        optimizer_u.step()
        optimizer_i.step()
        
        return model_client, loss.item()

    def aggregate_clients_params(self, round_user_params):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        
        if self.config['construct_graph_source'] == 'item':
            user_relation_graph = construct_user_relation_graph_via_item(round_user_params, self.config['num_items'],
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
        else:
            user_relation_graph = construct_user_relation_graph_via_user(round_user_params,
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
        
        # select the top-k neighborhood for each user.
        topk_user_relation_graph = select_topk_neighboehood(user_relation_graph, self.config['neighborhood_size'],
                                                            self.config['neighborhood_threshold'])
        # update item embedding via message passing.
        updated_item_embedding = MP_on_graph(round_user_params, self.config['num_items'], self.config['latent_dim'],
                                             topk_user_relation_graph, self.config['mp_layers'])
        self.server_model_param['embedding_item.weight'] = copy.deepcopy(updated_item_embedding)

    def aggregate_clients_params_user(self, round_user_params,round):
        """receive client models' parameters in a round, aggregate them and store the aggregated result for server."""
        # construct the user relation graph via embedding similarity.
        if round%self.config['update_round'] == 0:
                self.user_relation_graph = construct_user_relation_graph_via_user(round_user_params, 
                                                            self.config['latent_dim'],
                                                            self.config['similarity_metric'])
        topk_user_relation_graph = select_topk_neighboehood(self.user_relation_graph, self.config['neighborhood_size'],
                                                            self.config['neighborhood_threshold'])
        # update item embedding via message passing.
        if self.config['alias'] == 'UFGraphFR-pre':
            updated_item_embedding = MP_on_graph_with_embedding_user(round_user_params, self.config['num_items'], self.config['latent_dim'],
                                                 topk_user_relation_graph, self.config['mp_layers'])
            self.server_model_param['embedding_user.weight'] = copy.deepcopy(updated_item_embedding)
        else:
            updated_item_embedding = MP_on_graph(round_user_params, self.config['num_items'], self.config['latent_dim'],
                                                topk_user_relation_graph, self.config['mp_layers'])
            self.server_model_param['embedding_item.weight'] = copy.deepcopy(updated_item_embedding)
            


    def fed_train_a_round(self, all_train_data, round_id, embeddingUtils):
        """train a round."""
        # sample users participating in single round.
        num_participants = int(self.config['num_users'] * self.config['clients_sample_ratio'])
        participants = random.sample(range(self.config['num_users']), num_participants)
        # store users' model parameters of current round.
        round_participant_params = {}

        # initialize server parameters for the first round.
        if round_id == 0:
            #self.server_model_param['embedding_item.weight'] = {}
            #for user in participants:
            #    self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
            if self.config['alias'] == 'UFGraphFR-pre':
                self.server_model_param['embedding_user.weight'] = {}
                for user in participants:
                    self.server_model_param['embedding_user.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_user.weight'].data.cpu())
                self.server_model_param['embedding_user.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
            else:
                self.server_model_param['embedding_item.weight'] = {}
                for user in tqdm.tqdm(participants, desc="Initializing Server Parameters", leave=False):
                    self.server_model_param['embedding_item.weight'][user] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())
                self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())



            # self.server_model_param['embedding_item.weight']['global'] = copy.deepcopy(self.model.state_dict()['embedding_item.weight'].data.cpu())

        # perform model updating for each participated user.
        user_embeddings = {}

        for user in tqdm.tqdm(participants, desc="Training Round {}:".format(round_id + 1), leave=False):
            # copy the client model architecture from self.model
            model_client = copy.deepcopy(self.model)
            # for the first round, client models copy initialized parameters directly.
            # for other rounds, client models receive updated user embedding and aggregated item embedding from server
            # and use local updated mlp parameters from last round.
            if self.config['use_jointembedding']:
                if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
                    if user not in user_embeddings.keys():
                        user_embeddings[user] = embeddingUtils.embedding_users(user)
            else:
                user_embeddings[user] = None
        
            if round_id != 0:
                # for participated users, load local updated parameters.
                user_param_dict = copy.deepcopy(self.model.state_dict())
                if self.config['alias'] == 'UFGraphFR-pre':
                    if user in self.client_model_params.keys():
                        for key in self.client_model_params[user].keys():
                            if self.config['use_cuda'] is True:
                                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                            else:
                                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
                    if self.config['use_cuda'] is True:
                        user_param_dict['embedding_user.weight'] = copy.deepcopy(self.server_model_param['embedding_user.weight'][user].data).cuda()
                    else:
                        user_param_dict['embedding_user.weight'] = copy.deepcopy(self.server_model_param['embedding_user.weight'][user].data)
                else:
                    if user in self.client_model_params.keys():
                        for key in self.client_model_params[user].keys():
                            if self.config['use_cuda'] is True:
                                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()
                            elif self.config['use_mps'] is True :
                                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to('mps')
                            else:
                                user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)
                    if self.config['use_cuda'] is True:
                        user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data).cuda()
                    elif self.config['use_mps'] is True :
                        user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data).to('mps')
                    else:
                        user_param_dict['embedding_item.weight'] = copy.deepcopy(self.server_model_param['embedding_item.weight'][user].data)


                model_client.load_state_dict(user_param_dict)
            # Defining optimizers
            # optimizer is responsible for updating mlp parameters.
            optimizer = None
            optimizer_t = None


            if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot' :
                base_params = [
                      {"params": model_client.user_mlp.parameters()},
                      {"params": model_client.fc_layers.parameters()},
                     {"params": model_client.affine_output.parameters()},
                     ]
                if self.config['use_transfermer']:
                    base_params.append({"params": model_client.multheadAttention_layer.parameters()})
                
                optimizer = torch.optim.SGD(
                     base_params,
                    lr=self.config['lr'])
            else:
                optimizer = torch.optim.SGD(
                        [{"params": model_client.fc_layers.parameters()}, 
                        {"params": model_client.affine_output.parameters()}],

                        lr=self.config['lr'])            
            # optimizer_i is responsible for updating item embedding.
            optimizer_i = torch.optim.SGD(model_client.embedding_item.parameters(),
                                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                                             self.config['lr'])  # Item optimizer\
            optimizers = []

            if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
                # optimizer_t = torch.optim.SGD(model_client.embedding_time.parameters(),
                #                          lr=self.config['lr'] * self.config['num_items'] * self.config['lr_eta'] -
                #                             self.config['lr'])  # Item optimizer\
                optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])
                
                optimizers = [optimizer, optimizer_i, optimizer_u]
               
            else:
                optimizer_u = torch.optim.SGD(model_client.embedding_user.parameters(),
                                          lr=self.config['lr'] / self.config['clients_sample_ratio'] * self.config[
                                              'lr_eta'] - self.config['lr'])  # User optimizer
                optimizers = [optimizer, optimizer_u, optimizer_i]
            # load current user's training data and instance a train loader.
            user_train_data = [all_train_data[0][user], all_train_data[1][user], all_train_data[2][user]]
            user_dataloader = self.instance_user_train_loader(user_train_data)
            model_client.train()
            # update client model.
            for epoch in range(self.config['local_epoch']):
                for batch_id, batch in enumerate(user_dataloader):
                    assert isinstance(batch[0], torch.LongTensor)
                    loss = None
                    if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':

                        model_client, loss = self.fed_train_single_batch_my(model_client, batch, optimizers, user, user_embeddings[user])
                    else:
                        model_client, loss = self.fed_train_single_batch(model_client, batch, optimizers, user)
            # print('[User {}]'.format(user))
            # obtain client model parameters.
            client_param = model_client.state_dict()
            # store client models' user embedding using a dict.
            self.client_model_params[user] = copy.deepcopy(client_param)
            for key in self.client_model_params[user].keys():
                self.client_model_params[user][key] = self.client_model_params[user][key].data.cpu()
            # round_participant_params[user] = copy.deepcopy(self.client_model_params[user])
            # del round_participant_params[user]['embedding_user.weight']
            round_participant_params[user] = {}

            round_participant_params[user]['embedding_item.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_item.weight'])
            if self.config['dp'] > 0:
                round_participant_params[user]['embedding_item.weight'] = round_participant_params[user]['embedding_item.weight'].view(-1)
            #round_participant_params[user]['embedding_item.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_item.weight'].shape).sample()
            if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
                round_participant_params[user]['embedding_user.weight'] = copy.deepcopy(self.client_model_params[user]['embedding_user.weight'])
                round_participant_params[user]['embedding_user.weight'] = round_participant_params[user]['embedding_user.weight'].view(-1)
                if self.config['dp'] > 0:
                    round_participant_params[user]['embedding_user.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_user.weight'].shape).sample()

                #round_participant_params[user]['embedding_user.weight'] += Laplace(0, self.config['dp']).expand(round_participant_params[user]['embedding_user.weight'].shape).sample()
        # aggregate client models in server side.
        if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
            self.aggregate_clients_params_user(round_participant_params,round=round_id)
        else:
            self.aggregate_clients_params(round_participant_params)
        return participants

    def fed_evaluate(self, evaluate_data,embeddingUtils):
        # evaluate all client models' performance using testing data.
        test_users, test_items = evaluate_data[0], evaluate_data[1]
        negative_users, negative_items = evaluate_data[2], evaluate_data[3]
        
        # ratings for computing loss.
        temp = [0] * 100
        temp[0] = 1
        ratings = torch.FloatTensor(temp)
        if self.config['use_cuda'] is True:
            test_users = test_users.cuda()
            test_items = test_items.cuda()
            negative_users = negative_users.cuda()
            negative_items = negative_items.cuda()
            ratings = ratings.cuda()
        if self.config['use_mps'] is True:
            test_users = test_users.to(torch.device("mps"))
            test_items = test_items.to(torch.device("mps"))
            negative_users = negative_users.to(torch.device("mps"))
            negative_items = negative_items.to(torch.device("mps"))
            ratings = ratings.to(torch.device("mps"))
        # store all users' test item prediction score.
        test_scores = None
        # store all users' negative items prediction scores.
        negative_scores = None
        all_loss = {}
        user_embeddings = {}
        for user in tqdm.tqdm(range(self.config['num_users']), desc="Evaluating...", leave=False):
            # load each user's mlp parameters.
            user_model = copy.deepcopy(self.model)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            if self.config['use_jointembedding']:
                if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
                    user_embeddings[user] = embeddingUtils.embedding_users(user)
            else:
                user_embeddings[user] = None
            
            if user in self.client_model_params.keys():
                for key in self.client_model_params[user].keys():
                    if self.config['use_cuda'] is True:
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).cuda()                        
                    elif self.config['use_mps'] is False:
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data).to(torch.device("mps"))
                    else:
                        user_param_dict[key] = copy.deepcopy(self.client_model_params[user][key].data)

                #user_param_dict['embedding_item.weight'] = copy.deepcopy(
                #    self.server_model_param['embedding_item.weight']['global'].data).cuda() if self.config['use_cuda'] is True else copy.deepcopy(
                #    self.server_model_param['embedding_item.weight']['global'].data)
            user_model.load_state_dict(user_param_dict)
            user_model.eval()
            with torch.no_grad():
                # obtain user's positive test information.
                test_user = test_users[user: user + 1]
                test_item = test_items[user: user + 1]


                # obtain user's negative test information.
                negative_user = negative_users[user * 99: (user + 1) * 99]
                negative_item = negative_items[user * 99: (user + 1) * 99]

                # perform model prediction.
                test_score = None
                negative_score = None

                if self.config['alias'] == 'UFGraphFR' or self.config['alias'] == 'UFGraphFR-wot':
                    test_item = [test_item]
                    negative_item = [negative_item]
                    test_score = user_model(test_item,user_embeddings[user])
                    negative_score = user_model(negative_item,user_embeddings[user])
                else:
                    test_score = user_model(test_user)
                    negative_score = user_model(negative_user)
                if user == 0:
                    test_scores = test_score
                    negative_scores = negative_score
                else:
                    test_scores = torch.cat((test_scores, test_score))
                    negative_scores = torch.cat((negative_scores, negative_score))
                ratings_pred = torch.cat((test_score, negative_score))
                loss = self.crit(ratings_pred.view(-1), ratings)
            all_loss[user] = loss.item()
        if self.config['use_cuda'] is True:
            test_users = test_users.cpu()
            test_items = test_items.cpu()
            test_scores = test_scores.cpu()
            negative_users = negative_users.cpu()
            negative_items = negative_items.cpu()
            negative_scores = negative_scores.cpu()
        if self.config['use_mps'] is True:
            test_users = test_users.to(torch.device("mps"))
            test_items = test_items.to(torch.device("mps"))
            test_scores = test_scores.to(torch.device("mps"))
            negative_users = negative_users.to(torch.device("mps"))
            negative_items = negative_items.to(torch.device("mps"))
            negative_scores = negative_scores.to(torch.device("mps"))
        self._metron.subjects = [test_users.data.view(-1).tolist(),
                                 test_items.data.view(-1).tolist(),
                                 test_scores.data.view(-1).tolist(),
                                 negative_users.data.view(-1).tolist(),
                                 negative_items.data.view(-1).tolist(),
                                 negative_scores.data.view(-1).tolist()]
        hit_ratio, ndcg = self._metron.cal_hit_ratio(), self._metron.cal_ndcg()
        return hit_ratio, ndcg, all_loss

    def save(self, alias, epoch_id, hit_ratio, ndcg):
        assert hasattr(self, 'model'), 'Please specify the exact model !'
        model_dir = self.config['model_dir'].format(alias, epoch_id, hit_ratio, ndcg)
        save_checkpoint(self.model, model_dir)
