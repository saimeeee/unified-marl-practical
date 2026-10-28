import torch
import math
import numpy as np
import transformers
import logging
get_logger = lambda: logging.getLogger('unified_marl.reward')
from unified_marl.algorithms.kou.reward_models.MR import MR
from unified_marl.algorithms.kou.reward_models.NMR import NMR
from unified_marl.algorithms.kou.reward_models.lstm import LSTMRewardModel
from unified_marl.algorithms.kou.reward_models.PrefTransformer import PrefTransformer
from unified_marl.algorithms.kou.reward_models.trajectory_gpt2 import TransRewardModel
from unified_marl.algorithms.kou.reward_models.MultiPrefTransformer import MultiPrefTransformer
from unified_marl.algorithms.kou.reward_models.encoder_decoder_divide import MultiTransRewardDivideModel
from unified_marl.algorithms.kou.reward_models.q_function import FullyConnectedQFunction
from tqdm import tqdm
logger = get_logger()

class RewardModel:
    def __init__(self, args):
        self.args = args

        # configs for reward model
        self.n_agents = args.n_agents
        self.n_actions = args.n_actions
        self.obs_shape = args.obs_shape
        self.action_type = args.action_type
        self.max_traj_length = args.max_traj_length
        self.device = 'cuda' if args.use_cuda else 'cpu'
        self.model_type = args.model_type
        self.pi_accuracy = args.pi_accuracy

        self.multi_transformer_config = MultiPrefTransformer.get_default_config()
        self.transformer_config = PrefTransformer.get_default_config()
        self.MR_config = MR.get_default_config()
        self.NMR_config = NMR.get_default_config()

        self.activations = 'relu'
        self.activation_final = 'none'
        self.orthogonal_init = 'False'

        self.train_epoch = args.reward_train_epoch

        if self.model_type == "MultiPrefTransformerDivide":
                config = transformers.GPT2Config(**self.multi_transformer_config)
                # config multi-transformer reward model
                trans = MultiTransRewardDivideModel(
                    config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, n_agent=self.n_agents,
                    action_type=self.action_type, max_episode_steps=args.max_traj_length, device=args.device,
                )
                # config model wrapper for train and eval
                self.reward_model = MultiPrefTransformer(config, trans, self.device)
        elif self.model_type == "PrefTransformer":
            config = transformers.GPT2Config(**self.transformer_config)
            # config transformer reward model
            trans = TransRewardModel(
                config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                activation=self.activations, activation_final=self.activation_final,
                max_episode_steps=self.max_traj_length, device=self.device,
            )
            # config model wrapper for train and eval
            self.reward_model = PrefTransformer(config, trans, self.device)
        elif self.model_type == "MR":
            rf = FullyConnectedQFunction(
                observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                inner_dim=self.MR_config.inner_dim, action_embd_dim=self.MR_config.action_embd_dim,
                orthogonal_init=self.orthogonal_init, activations=self.activations,
                activation_final=self.activation_final, device=self.device,
            )
            self.reward_model = MR(self.MR_config, rf, self.device)
        elif self.model_type == "NMR":
            config = transformers.GPT2Config(**self.NMR_config)
            lstm = LSTMRewardModel(
                config=config, observation_dim=self.obs_shape, action_dim=self.n_actions, action_type=self.action_type,
                activation=self.activations, activation_final=self.activation_final,
                max_episode_steps=self.max_traj_length, device=self.device,
            )
            self.reward_model = NMR(config, lstm, self.device)
    
    def train(self, pref_dataset):
        train_batch = {}
        eval_batch = {}
        eval_mini_batch = {}
        mask = pref_dataset['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - pref_dataset["terminated"][:, :-1])
        train_bs = int(self.args.reward_batch_size / 2)
        eval_bs = int(pref_dataset['obs'].shape[0] / 2)
        interval = int(eval_bs / train_bs) + 1
        eval_batch['observations0'] = pref_dataset['obs'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['observations1'] = pref_dataset['obs'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['actions0'] = pref_dataset['action'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['actions1'] = pref_dataset['action'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['masks0'] = mask[:eval_bs, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['masks1'] = mask[eval_bs:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['labels'] = pref_dataset['labels'].to(self.device)
        eval_batch['labels'] = torch.cat([1 - eval_batch['labels'], eval_batch['labels']], dim=-1)
        keys = ['observations0', 'observations1', 'actions0', 'actions1', 'timesteps0', 'timesteps1', 'masks0', 'masks1', 'labels']
        for i in range(self.train_epoch):
            shuffled_idx = np.random.permutation(eval_bs)
            for j in range(interval):
                start_pt0 = j * train_bs
                end_pt0 = min((j + 1) * train_bs, eval_bs)
                sample_number0 = shuffled_idx[start_pt0: end_pt0]
                sample_number1 = sample_number0 + eval_bs
                train_batch['observations0'] = pref_dataset['obs'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['actions0'] = pref_dataset['action'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['agent_ids0'] = pref_dataset['agent_ids'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['observations1'] = pref_dataset['obs'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['actions1'] = pref_dataset['action'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['agent_ids1'] = pref_dataset['agent_ids'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['masks0'] = mask[sample_number0][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['masks1'] = mask[sample_number1][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['labels'] = pref_dataset['labels'][sample_number0].to(self.device)
                train_batch['labels'] = torch.cat([1 - train_batch['labels'], train_batch['labels']], dim=-1)
                metrics = self.reward_model.train(train_batch)
            acc_total = 0
            for k in range(interval):
                start_pt0 = k * train_bs
                end_pt0 = min((k + 1) * train_bs, eval_bs)
                sample_number = shuffled_idx[start_pt0: end_pt0]
                for key in keys:
                    eval_mini_batch[key] = eval_batch[key][sample_number]
                acc_batch = self.reward_model.evaluation_acc(eval_mini_batch)
                acc_total += acc_batch
            # print(acc)
            acc = acc_total/interval
            logger.info('acc:%f', acc)
            if acc >= 0.97:
                break
        preds0, preds1 = [], []
        indices = np.arange(eval_bs)
        for l in range(interval):
            start_pt0 = l * train_bs
            end_pt0 = min((l + 1) * train_bs, eval_bs)
            sample_number = indices[start_pt0: end_pt0]
            for key in keys:
                eval_mini_batch[key] = eval_batch[key][sample_number]
            preds0_batch, preds1_batch = self.reward_model.get_reward_for_offline(eval_mini_batch)
            preds0.append(preds0_batch)
            preds1.append(preds1_batch)
        return torch.cat(preds0 + preds1, dim=0)
    
    def train_predicter(self, pref_dataset):
        train_batch = {}
        eval_batch = {}
        eval_mini_batch = {}
        mask = pref_dataset['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - pref_dataset["terminated"][:, :-1])
        train_bs = int(self.args.reward_batch_size / 2)
        eval_bs = int(pref_dataset['obs'].shape[0] / 2)
        interval = int(eval_bs / train_bs) + 1
        indices = np.arange(eval_bs)
        eval_batch['observations0'] = pref_dataset['obs'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['observations1'] = pref_dataset['obs'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['actions0'] = pref_dataset['action'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['actions1'] = pref_dataset['action'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['masks0'] = mask[:eval_bs, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['masks1'] = mask[eval_bs:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['labels'] = pref_dataset['labels'].to(self.device)
        eval_batch['labels'] = torch.cat([1 - eval_batch['labels'], eval_batch['labels']], dim=-1)
        keys = ['observations0', 'observations1', 'actions0', 'actions1', 'timesteps0', 'timesteps1', 'masks0', 'masks1', 'labels']
        for i in range(self.train_epoch):
            shuffled_idx = np.random.permutation(eval_bs)
            for j in range(interval):
                start_pt0 = j * train_bs
                end_pt0 = min((j + 1) * train_bs, eval_bs)
                sample_number0 = shuffled_idx[start_pt0: end_pt0]
                sample_number1 = sample_number0 + eval_bs
                train_batch['observations0'] = pref_dataset['obs'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['actions0'] = pref_dataset['action'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['agent_ids0'] = pref_dataset['agent_ids'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['observations1'] = pref_dataset['obs'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['actions1'] = pref_dataset['action'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['agent_ids1'] = pref_dataset['agent_ids'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['masks0'] = mask[sample_number0][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['masks1'] = mask[sample_number1][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['labels'] = pref_dataset['labels'][sample_number0].to(self.device)
                train_batch['labels'] = torch.cat([1 - train_batch['labels'], train_batch['labels']], dim=-1)
                metrics = self.reward_model.train(train_batch)
                # logger.info('trans:%f, pi:%f, loss:%f', metrics['trans_loss'], metrics['pi_loss'], metrics['loss'])
            acc_total = 0
            T = 1
            preds0, preds1, pred= [], [], []
            for k in range(interval):
                start_pt0 = k * train_bs
                end_pt0 = min((k + 1) * train_bs, eval_bs)
                sample_number = indices[start_pt0: end_pt0]
                for key in keys:
                    eval_mini_batch[key] = eval_batch[key][sample_number]
                acc, preds0_batch, preds1_batch= self.reward_model.evaluation_agent_acc(eval_mini_batch)
                acc_total += acc
                with torch.no_grad():
                    preds0.append(preds0_batch)
                    preds1.append(preds1_batch)
            pred = torch.cat(preds0 + preds1, dim=0).sum(1).squeeze(-1)
            # print(acc)
            pi_returns = torch.softmax(pred / T, dim=-1)
            agent_ids = pref_dataset['agent_ids'].to('cuda')
            pi = (pi_returns * agent_ids).sum()/pref_dataset['obs'].shape[0]
            logger.info('acc:%f, pi:%f', acc_total/interval, pi)
            if acc_total/interval >= 0.97 and pi >= self.pi_accuracy:
                break
            
        return torch.cat(preds0 + preds1, dim=0)
    
    def predict_for_unlabeled_dataset_using_predictor(self, pref_dataset):
        eval_batch = {}
        mask = pref_dataset['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - pref_dataset["terminated"][:, :-1])
        eval_bs = int(pref_dataset['obs'].shape[0] / 2)
        eval_batch['observations0'] = pref_dataset['obs'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['observations1'] = pref_dataset['obs'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['actions0'] = pref_dataset['action'][:eval_bs, :self.max_traj_length].to(self.device)
        eval_batch['actions1'] = pref_dataset['action'][eval_bs:, :self.max_traj_length].to(self.device)
        eval_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
        eval_batch['masks0'] = mask[:eval_bs, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        eval_batch['masks1'] = mask[eval_bs:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
        preds0, preds1= self.reward_model.get_agent_reward_for_offline(eval_batch)
        return torch.cat([preds0, preds1], dim=0)
    
    def e2e_rm_forward(self, batch):
        eval_batch = {}
        mask = batch['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - batch["terminated"][:, :-1])
        eval_batch['obs'] = batch['obs']
        eval_batch['actions'] = batch['actions']
        eval_batch['timesteps'] = torch.arange(0,batch["obs"].shape[1]).unsqueeze(0).repeat(batch["obs"].shape[0], 1).to(self.device)
        eval_batch['masks'] = mask[:batch["obs"].shape[0], :batch["obs"].shape[1]].repeat(1, 1, self.n_agents).to(self.device)
        preds0 = self.reward_model.get_batch_reward(eval_batch)

        return preds0

    def get_threshold_beta(self, update_step):
        return max(1, -(3-1)/30 * update_step + 3)

    def train_diverse_predictor(self, pref_dataset):
        batch, train_batch = {}, {}
        mask = pref_dataset['filled'].float()
        mask[:, 1:] = mask[:, 1:] * (1 - pref_dataset["terminated"][:, :-1])
        train_bs = int(self.args.reward_batch_size / 2)
        eval_bs = int(pref_dataset['obs'].shape[0] / 2)
        keys = ['observations0', 'observations1', 'actions0', 'actions1', 'timesteps0', 'timesteps1', 'masks0', 'masks1', 'labels']
        for i in tqdm(range(self.train_epoch), desc="Training Epochs"):
            batch['obs0'] = pref_dataset['obs'][:eval_bs][:, :self.max_traj_length].to(self.device)
            batch['action0'] = pref_dataset['action'][:eval_bs][:, :self.max_traj_length].to(self.device)
            batch['obs1'] = pref_dataset['obs'][eval_bs:][:, :self.max_traj_length].to(self.device)
            batch['action1'] = pref_dataset['action'][eval_bs:][:, :self.max_traj_length].to(self.device)
            batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
            batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(eval_bs, 1).to(self.device)
            batch['masks0'] = mask[:eval_bs][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
            batch['masks1'] = mask[eval_bs:][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
            batch['global_labels'] = pref_dataset['global_labels'][:eval_bs].to(self.device)
            batch['global_labels'] = torch.cat([1 - batch['global_labels'], batch['global_labels']], dim=-1)
            batch['between_labels'] = pref_dataset['between_labels'][:eval_bs].to(self.device)
            batch['between_labels'] = torch.stack([1 - batch['between_labels'], batch['between_labels']], dim=-1)
            batch['in_labels0'] = pref_dataset['in_labels'][0][:eval_bs].to(self.device)
            batch['in_labels1'] = pref_dataset['in_labels'][1][:eval_bs].to(self.device)
            global_KL_div, between_KL_div, in_KL_div = self.reward_model.get_KL(batch)
            # filter trust samples according to KLs
            # if i==0, 所有baseline都取一个很大的值
            if i == 0:
                global_baseline = 1000.0
                between_baseline = 1000.0
                in_baseline = 1000.0
                trust_index = (global_KL_div < global_baseline) & (between_KL_div < between_baseline) & (in_KL_div < in_baseline)
            else:
                global_baseline = -torch.log(global_KL_div.max() + 1e-8) + 0.5 * global_KL_div.max() + min(self.get_threshold_beta(i) * global_KL_div.var(), 3.0)
                between_baseline = -torch.log(between_KL_div.max() + 1e-8) + 0.5 * between_KL_div.max() + min(self.get_threshold_beta(i) * between_KL_div.var(), 3.0)
                in_baseline = -torch.log(in_KL_div.max() + 1e-8) + 0.5 * in_KL_div.max() + min(self.get_threshold_beta(i) * in_KL_div.var(), 3.0)
                trust_index = (global_KL_div < global_baseline) & (between_KL_div < between_baseline) & (in_KL_div < in_baseline)
            trust_indices = torch.where(trust_index)[0].tolist()
            interval = math.ceil(len(trust_indices) / train_bs)
            print("len",len(trust_indices))
            for j in tqdm(range(interval), desc="Training Intervals"):
                # filter samples according to KL_divs
                shuffled_idx = np.random.permutation(trust_indices)
                start_pt0 = j * train_bs
                end_pt0 = min((j + 1) * train_bs, len(trust_indices))
                sample_number0 = shuffled_idx[start_pt0: end_pt0]
                sample_number1 = sample_number0 + eval_bs
                train_batch['obs0'] = pref_dataset['obs'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['action0'] = pref_dataset['action'][sample_number0][:, :self.max_traj_length].to(self.device)
                train_batch['obs1'] = pref_dataset['obs'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['action1'] = pref_dataset['action'][sample_number1][:, :self.max_traj_length].to(self.device)
                train_batch['timesteps0'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['timesteps1'] = torch.arange(0,self.max_traj_length).unsqueeze(0).repeat(end_pt0 - start_pt0, 1).to(self.device)
                train_batch['masks0'] = mask[sample_number0][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['masks1'] = mask[sample_number1][:, :self.max_traj_length].repeat(1, 1, self.n_agents).to(self.device)
                train_batch['global_labels'] = pref_dataset['global_labels'][sample_number0].to(self.device)
                train_batch['global_labels'] = torch.cat([1 - train_batch['global_labels'], train_batch['global_labels']], dim=-1)
                train_batch['between_labels'] = pref_dataset['between_labels'][sample_number0].to(self.device)
                train_batch['between_labels'] = torch.stack([1 - train_batch['between_labels'], train_batch['between_labels']], dim=-1)
                train_batch['in_labels0'] = pref_dataset['in_labels'][0][sample_number0].to(self.device)
                train_batch['in_labels1'] = pref_dataset['in_labels'][1][sample_number0].to(self.device)
                metrics = self.reward_model.train_diverse(train_batch)
                # todo: log metrics loss sum
                total_loss, global_loss, between_loss, in_loss = metrics['loss'], metrics['trans_loss'], metrics['between_loss'], metrics['in_loss']
        
        preds0, preds1 = self.reward_model.get_agent_reward_for_offline(batch)

        return torch.cat([preds0, preds1], dim=0)