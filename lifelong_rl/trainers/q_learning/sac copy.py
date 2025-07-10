import numpy as np
import torch
import torch.optim as optim
from torch import nn as nn

from collections import OrderedDict

import lifelong_rl.torch.pytorch_util as ptu
from lifelong_rl.torch.distributions import TanhNormal
from lifelong_rl.util.eval_util import create_stats_ordered_dict
from lifelong_rl.core.rl_algorithms.torch_rl_algorithm import TorchTrainer
from lifelong_rl.torch.pytorch_util import np_to_pytorch_batch

# from torch_batch_svd import svd

ACTION_MIN = -1.0
ACTION_MAX = 1.0


class SACTrainer(TorchTrainer):
    """
    Soft Actor Critic (Haarnoja et al. 2018). (Offline training ver.)
    Continuous maximum Q-learning algorithm with parameterized actor.
    """
    def __init__(
            self,
            env,  # Associated environment for learning
            policy,  # Associated policy (should be TanhGaussian)
            qfs,  # Q functions
            target_qfs,  # Slow updater to Q functions
            discount=0.99,  # Discount factor
            reward_scale=1.0,  # Scaling of rewards to modulate entropy bonus
            use_automatic_entropy_tuning=True,  # Whether to use the entropy-constrained variant
            target_entropy=None,  # Target entropy for entropy-constraint variant
            policy_lr=3e-4,  # Learning rate of policy and entropy weight
            qf_lr=3e-4,  # Learning rate of Q functions
            optimizer_class=optim.Adam,  # Class of optimizer for all networks
            soft_target_tau=5e-3,  # Rate of update of target networks
            target_update_period=1,  # How often to update target networks
            max_q_backup=False,
            deterministic_backup=False,
            policy_eval_start=0,
            eta=-1.0,

            num_qs=10,
            q_samples=4,
            replay_buffer=None,
    ):
        super().__init__()

        self.env = env
        self.policy = policy
        self.qfs = qfs
        self.target_qfs = target_qfs
        self.num_qs = num_qs
        self.q_samples = q_samples

        self.discount = discount
        self.reward_scale = reward_scale
        self.soft_target_tau = soft_target_tau
        self.target_update_period = target_update_period

        self.max_q_backup = max_q_backup
        self.deterministic_backup = deterministic_backup
        self.eta = eta

        self.replay_buffer = replay_buffer

        self.use_automatic_entropy_tuning = use_automatic_entropy_tuning
        if self.use_automatic_entropy_tuning:
            if target_entropy:
                self.target_entropy = target_entropy
            else:
                # Heuristic value: dimension of action space
                self.target_entropy = -np.prod(
                    self.env.action_space.shape).item()
            self.log_alpha = ptu.zeros(1, requires_grad=True)
            self.alpha_optimizer = optimizer_class(
                [self.log_alpha],
                lr=policy_lr,
            )

        self.qf_criterion = nn.MSELoss(reduction='none')

        self.policy_optimizer = optimizer_class(
            self.policy.parameters(),
            lr=policy_lr,
        )
        self.qfs_optimizer = optimizer_class(
            self.qfs.parameters(),
            lr=qf_lr,
        )

        self.eval_statistics = OrderedDict()
        self._need_to_update_eval_statistics = True
        self.policy_eval_start = policy_eval_start

    def _get_tensor_values(self, obs, actions, network=None):
        action_shape = actions.shape[0]
        obs_shape = obs.shape[0]
        num_repeat = int(action_shape / obs_shape)
        obs_temp = obs.unsqueeze(1).repeat(1, num_repeat,
                                           1).view(obs.shape[0] * num_repeat,
                                                   obs.shape[1])
        preds, _ = network(obs_temp, actions)
        preds = preds.view(-1, obs.shape[0], num_repeat, 1)
        return preds

    def _get_policy_actions(self, obs, num_actions, network=None):
        obs_temp = obs.unsqueeze(1).repeat(1, num_actions,
                                           1).view(obs.shape[0] * num_actions,
                                                   obs.shape[1])
        new_obs_actions, _, _, new_obs_log_pi, *_ = network(
            obs_temp,
            reparameterize=True,
            return_log_prob=True,
        )
        return new_obs_actions.detach(), new_obs_log_pi.view(
            obs.shape[0], num_actions, 1).detach()

    def train_from_torch(self, batch, indices):
        obs= batch['observations']
        next_obs = batch['next_observations']
        actions = batch['actions']
        rewards = batch['rewards']
        terminals = batch['terminals']
        
        if self.eta > 0:
            actions.requires_grad_(True)
        
        """
        Policy and Alpha Loss
        """

        new_obs_actions, policy_mean, policy_log_std, log_pi, *_ = self.policy(
            obs,
            reparameterize=True,
            return_log_prob=True,
        )

        if self.use_automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha *
                           (log_pi + self.target_entropy).detach()).mean()
            alpha = self.log_alpha.exp()
        else:
            alpha_loss = 0
            alpha = 1

        q_new_actions = self.qfs.sample(obs, new_obs_actions)

        policy_loss = (alpha * log_pi - q_new_actions).mean()

        if self._num_train_steps < self.policy_eval_start:
            """
            For the initial few epochs, try doing behaivoral cloning, if needed
            conventionally, there's not much difference in performance with having 20k 
            gradient steps here, or not having it
            """
            policy_log_prob = self.policy.get_log_probs(obs.detach(), actions)
            policy_loss = (alpha * log_pi - policy_log_prob).mean()

            
        """
        QF Loss
        """
        # (num_qs, batch_size, output_size)
        qs_pred, _ = self.qfs(obs, actions)

        new_next_actions, _, _, new_log_pi, *_ = self.policy(
            next_obs,
            reparameterize=False,
            return_log_prob=True,
        )

        if not self.max_q_backup:
            target_q_values = self.target_qfs.sample(next_obs, new_next_actions)
            if not self.deterministic_backup:
                target_q_values -= alpha * new_log_pi
        else:
            # if self.max_q_backup
            next_actions_temp, _ = self._get_policy_actions(
                next_obs, num_actions=10, network=self.policy)
            target_q_values = self._get_tensor_values(
                next_obs, next_actions_temp,
                network=self.qfs).max(2)[0].min(0)[0]

        future_values = (1. - terminals) * self.discount * target_q_values
        q_target = self.reward_scale * rewards + future_values

        qfs_loss = self.qf_criterion(qs_pred, q_target.detach().unsqueeze(0))
        qfs_loss = qfs_loss.mean(dim=(1, 2)).sum()

        qfs_loss_total = qfs_loss
        # if self.eta > 0:
        #     obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
        #     actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
        #     qs_preds_tile, _ = self.qfs(obs_tile, actions_tile)
        #     qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
        #     qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        #     qs_pred_grads = qs_pred_grads.transpose(0, 1)
            # print("qs_pred_grads", qs_pred_grads.shape)
            # grad_loss = self.estimate_entropy(qs_pred_grads)
            
            # qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            # masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            # qs_pred_grads = (1 - masks) * qs_pred_grads
            # grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)
            # print(grad_loss)
            # exit()
            # qfs_loss_total -= self.eta * grad_loss
        # if self.eta > 0:
        #     obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
        #     actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
        #     qs_preds_tile, sample_idx = self.qfs(obs_tile, actions_tile)
        #     idx = torch.tensor(sample_idx, device=ptu.device)

        #     # print(sample_idx)
        #     # print(idx)
        #     # exit(0)

        #     remaining = [x for x in range(self.num_qs) if x not in sample_idx]
        #     # print("actions_tile[idx]",actions_tile[idx].shape)
        #     qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            
        #     # print(qs_pred_grads[idx].shape)
        #     # print("qs_pred_grads",qs_pred_grads.shape)
        #     # exit()

        #     qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        #     # qs_pred_grads = qs_pred_grads.transpose(0, 1)
            

        #     # print("qs_pred_grads", qs_pred_grads.shape)
        #     # repeat_updata_num = 2 * self.q_samples - self.num_qs
        #     # a = sample_idx[:repeat_updata_num]
        #     # print(a)
        #     # exit()
        #     # pearson_loss = self.pearson_corr_custom_m(qs_pred_grads[sample_idx], sample_idx[:repeat_updata_num])
        #     # print(pearson_loss.shape)
        #     # exit()

        #     # 平均损失
        #     # grad_loss = torch.mean(cos_sim)
        #     # print(grad_loss.shape)
        #     qs_pred_grads = qs_pred_grads.transpose(0, 1)
        #     qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        #     # masks = self.create_tensor_mask(self.num_qs, sample_idx[0]).to(device=ptu.device)
        #     masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        #     # print(masks)
        #     # exit()
        #     # masks = masks.unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        #     # print(masks[0])
        #     # e_num = torch.eye(self.num_qs, device=ptu.device)
        #     # # e_num[:, remaining] = 1
        #     # masks = e_num.unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        #     # print(masks.shape)
        #     # print(qs_pred_grads.shape)
        #     # exit()
        #     # per_loss_2_0 = self.pearson_corr(ve_3, ve_1)  # shape [B]
        #     # per_loss_2_1 = self.pearson_corr(ve_3, ve_2)  # shape [B]
        #     # per_loss_3_0 = self.pearson_corr(ve_4, ve_1)  # shape [B]
        #     # per_loss_3_1 = self.pearson_corr(ve_4, ve_2)  # shape [B]

        #     # per_loss = torch.mean(per_loss_2_0 + per_loss_2_1 + per_loss_3_0 + per_loss_3_1) / (len(sample_idx) - 1)
        #     # pearson_loss_mean = torch.mean(torch.sum(pearson_loss, dim=(1, 2))) / (len(sample_idx) - 1)
        #     # print("pearson_loss",pearson_loss.shape)
        #     # print("pearson_loss_mean",pearson_loss_mean.shape)
        #     qs_pred_grads = (1-masks) * qs_pred_grads
        #     grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (len(sample_idx) - 1)
        #     # print("qs_pred_grads",qs_pred_grads.shape)
        #     # print("grad_loss",grad_loss.shape)
        #     # exit()

        #     # print("grad_loss + per_loss", grad_loss + per_loss)
        #     # print("grad_loss", grad_loss)
        #     # print("per_loss", per_loss)
            
        #     qfs_loss_total += self.eta * (grad_loss)
        
        # if self.eta > 0:
        #     obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
        #     actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
        #     qs_preds_tile, sample_idx = self.qfs(obs_tile, actions_tile)
        #     idx = torch.tensor(sample_idx, device=ptu.device)



        #     remaining = [x for x in range(self.num_qs) if x not in sample_idx]

        #     qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            


        #     qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)

        #     repeat_updata_num = 2 * self.q_samples - self.num_qs

        #     pearson_loss = self.pearson_corr_custom_m(qs_pred_grads[sample_idx], sample_idx[:repeat_updata_num])


        #     # 平均损失

        #     qs_pred_grads = qs_pred_grads.transpose(0, 1)
        #     qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        #     masks = self.create_tensor_mask(self.num_qs, sample_idx[0]).to(device=ptu.device)

        #     masks = masks.unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)

        #     pearson_loss_mean = torch.mean(torch.sum(pearson_loss, dim=(1, 2))) / (len(sample_idx) - 1)

        #     qs_pred_grads = masks * qs_pred_grads
        #     grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (len(sample_idx) - 1)

            
        #     qfs_loss_total += self.eta * (grad_loss + pearson_loss_mean)
            
        # if self.eta > 0:
        #     obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
        #     actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
        #     qs_preds_tile, sample_idx = self.qfs(obs_tile, actions_tile)
        #     idx = torch.tensor(sample_idx, device=ptu.device)

        #     # print(sample_idx)
        #     # print(idx)
        #     # exit(0)

        #     remaining = [x for x in range(self.num_qs) if x not in sample_idx]
        #     qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
        #     qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
        #     qs_pred_grads = qs_pred_grads.transpose(0, 1)
        #     X = qs_pred_grads
        #     # qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
        #     # masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        #     # qs_pred_grads = (1 - masks) * qs_pred_grads
        #     B, Q, D = X.shape
        #     X_centered = X - X.mean(dim=1, keepdim=True)  # [B, Q, D]

        #     # 计算协方差矩阵
        #     cov = torch.matmul(X_centered.transpose(1, 2), X_centered) / (Q - 1)  # [B, D, D]
        #     eps = 1e-6
        #     eye = torch.eye(cov.size(-1), device=cov.device).unsqueeze(0)
        #     cov = cov + eps * eye
        #     # print(cov.shape)
        #     # exit()
        #     # Step 2: Cholesky 分解并求逆矩阵（白化）
        #     L = torch.linalg.cholesky(cov)        # [B, D, D]
        #     # exit()
        #     L_inv = torch.linalg.inv(L)           # [B, D, D]

        #     # Step 3: 白化数据：X' = X @ L^{-1,T}
        #     # X: [B, Q, D] × L_inv.T: [B, D, D] → [B, Q, D]
        #     X_whitened = torch.matmul(X, L_inv.transpose(-1, -2))  # [B, Q, D]

        #     # Step 4: 计算余弦相似度矩阵（每个样本内）
        #     # [B, Q, D] × [B, D, Q] → [B, Q, Q]
        #     dot = torch.matmul(X_whitened, X_whitened.transpose(1, 2))  # [B, Q, Q]

        #     # 计算范数 [B, Q]
        #     norm = torch.norm(X_whitened, dim=-1)  # [B, Q]
        #     norm_matrix = torch.matmul(norm.unsqueeze(2), norm.unsqueeze(1))  # [B, Q, Q]

        #     sim_matrix = dot / (norm_matrix + 1e-8)  # 加eps防止除0
            # e_num = torch.eye(self.num_qs, device=ptu.device)
            # e_num[:, remaining] = 1
            # masks = e_num.unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
        #     sim_matrix = (1-masks) * sim_matrix
        #     # print(sim_matrix.shape)
        #     # exit()
        #     grad_loss = torch.mean(torch.sum(sim_matrix, dim=(1, 2))) / (self.num_qs - 1)
        #     # print(grad_loss)
        #     qfs_loss_total += self.eta * grad_loss
        if self.eta > 0:
            obs_tile = obs.unsqueeze(0).repeat(self.num_qs, 1, 1)
            actions_tile = actions.unsqueeze(0).repeat(self.num_qs, 1, 1).requires_grad_(True)
            qs_preds_tile = self.qfs(obs_tile, actions_tile)
            qs_pred_grads, = torch.autograd.grad(qs_preds_tile.sum(), actions_tile, retain_graph=True, create_graph=True)
            qs_pred_grads = qs_pred_grads / (torch.norm(qs_pred_grads, p=2, dim=2).unsqueeze(-1) + 1e-10)
            qs_pred_grads = qs_pred_grads.transpose(0, 1)
            
            qs_pred_grads = torch.einsum('bik,bjk->bij', qs_pred_grads, qs_pred_grads)
            masks = torch.eye(self.num_qs, device=ptu.device).unsqueeze(dim=0).repeat(qs_pred_grads.size(0), 1, 1)
            qs_pred_grads = (1 - masks) * qs_pred_grads
            grad_loss = torch.mean(torch.sum(qs_pred_grads, dim=(1, 2))) / (self.num_qs - 1)
            
            qfs_loss_total += self.eta * grad_loss

        if self.use_automatic_entropy_tuning and not self.deterministic_backup:
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.qfs_optimizer.zero_grad()
        qfs_loss_total.backward()
        self.qfs_optimizer.step()

        self.try_update_target_networks()
        """
        Save some statistics for eval
        """
        if self._need_to_update_eval_statistics:
            self._need_to_update_eval_statistics = False

            policy_loss = ptu.get_numpy(log_pi - q_new_actions).mean()
            policy_avg_std = ptu.get_numpy(torch.exp(policy_log_std)).mean()

            self.eval_statistics['QFs Loss'] = np.mean(
                ptu.get_numpy(qfs_loss)) / self.num_qs
            # if self.eta > 0:
            #     self.eval_statistics['Q Grad Loss'] = np.mean(
            #         ptu.get_numpy(grad_loss))
            self.eval_statistics['Policy Loss'] = np.mean(policy_loss)

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Predictions',
                    ptu.get_numpy(qs_pred),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Qs Targets',
                    ptu.get_numpy(q_target),
                ))

            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Log Pis',
                    ptu.get_numpy(log_pi),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy mu',
                    ptu.get_numpy(policy_mean),
                ))
            self.eval_statistics.update(
                create_stats_ordered_dict(
                    'Policy log std',
                    ptu.get_numpy(policy_log_std),
                ))
            self.eval_statistics['Policy std'] = np.mean(policy_avg_std)

            if self.use_automatic_entropy_tuning:
                self.eval_statistics['Alpha'] = alpha.item()
                self.eval_statistics['Alpha Loss'] = alpha_loss.item()

    def try_update_target_networks(self):
        if self._num_train_steps % self.target_update_period == 0:
            self.update_target_networks()

    def update_target_networks(self):
        ptu.soft_update_from_to(self.qfs, self.target_qfs,
                                self.soft_target_tau)

    def get_diagnostics(self):
        return self.eval_statistics

    def end_epoch(self, epoch):
        self._need_to_update_eval_statistics = True


    def select_ica_indices_and_pearson_loss_3d(self, x: torch.Tensor, m: int, random_state: int = 0):
        """
        参数:
            x: [n, 256, d] 的张量，表示 n 个样本，每个样本是 [256, d] 的矩阵
            m: 选取的样本数目
            random_state: FastICA 的随机种子
        返回:
            indices: 选中样本的原始索引
            pearson_loss: 选中样本的 Pearson 平方相关损失
        """
        if x.ndim != 3:
            raise ValueError("x 应该是 [n, 256, d] 的三维张量")

        n, t, d = x.shape
        if m > n:
            raise ValueError("m 不能大于 n")

        # Step 1: reshape 成 [n, 256*d] 以供 ICA 使用
        x_reshaped = x.reshape(n, t * d)
        x_np = x_reshaped.cpu().numpy()

        # Step 2: FastICA 选出 m 个代表性样本
        ica = FastICA(n_components=m, random_state=random_state)
        S = ica.fit_transform(x_np)  # shape: [n, m]

        # Step 3: 对每个成分选出 loading 最大的样本
        indices = np.abs(S).argmax(axis=0)
        # 避免重复
        taken = set()
        for j, idx in enumerate(indices):
            if idx in taken:
                order = np.argsort(-np.abs(S)[:, j])  # 绝对值从大到小
                for alt in order:
                    if alt not in taken:
                        idx = alt
                        break
            taken.add(idx)
            indices[j] = idx
        indices = indices.tolist()

        # Step 4: 提取选中的 m 个样本 [m, 256, d]
        selected = x[indices]  # [m, 256, d]

        # Step 5: 计算皮尔逊相关损失
        m = selected.shape[0]
        # flatten each [256, d] into a vector
        flattened = selected.reshape(m, -1)  # [m, 256*d]
        centered = flattened - flattened.mean(dim=1, keepdim=True)
        std = centered.std(dim=1, keepdim=True) + 1e-8
        normed = centered / std

        corr_matrix = normed @ normed.T / (flattened.shape[1] - 1)  # [m, m]
        # 只取非对角元素
        mask = ~torch.eye(m, dtype=torch.bool, device=x.device)
        pearson_loss = (corr_matrix[mask] ** 2).mean()

        return indices, pearson_loss



    def estimate_entropy(self, actions, num_components=3):  # (batch, sample, dim)
        batch_size, n_samples, dim = actions.shape
        # actions = actions.detach()
        diffs = actions.unsqueeze(2) - actions.unsqueeze(1)  # (B, S, S, D)
        dists = torch.norm(diffs, dim=-1)  # (B, S, S)
        dists = dists + 1e-6  # avoid log(0)
        entropy = torch.log(dists).mean(dim=(1, 2))  # batch mean entropy
        return entropy.mean()
    
    def create_tensor_mask(self, num: int, x: int) -> torch.Tensor:
        slide_w = self.q_samples * 2 - num
        mat = torch.zeros((num, num))
        indices = list(range(x, x + slide_w))

        for i in indices:
            for j in indices:
                if i != j:
                    mat[i][j] = 1
        

        # if x + slide_w - 1 < num:  # 防止越界
        #     mat[x, x + 1] = 1.0
        #     mat[x + 1, x] = 1.0

        return mat
    
    def pearson_corr(self, x, y):
        """
        x: [B, D]
        y: [B, D]
        return: [B] Pearson correlation per batch
        """
        x_centered = x - x.mean(dim=1, keepdim=True)
        y_centered = y - y.mean(dim=1, keepdim=True)
        numerator = torch.sum(x_centered * y_centered, dim=1)
        denominator = torch.norm(x_centered, dim=1) * torch.norm(y_centered, dim=1)
        return numerator / (denominator + 1e-8)
    

    def pearson_corr_custom_m(self, X, m_indices):
        """
        X: Tensor of shape [n, 256, 6]
        m_indices: List[int]，例如 [4, 5, 0, 1]，不要求连续或升序
        Return: Tensor of shape [n - m, m, 256]
        """
        n, B, D = X.shape
        # print("B", B)
        m_indices = torch.tensor(m_indices, dtype=torch.long, device=X.device)
        m = len(m_indices)

        # 构建剩余索引
        all_indices = torch.arange(n, device=X.device)
        mask = torch.ones(n, dtype=torch.bool, device=X.device)
        mask[m_indices] = False
        rest_indices = all_indices[mask]  # shape [n - m]

        # 提取 A 和 B 向量组
        X_A = X[m_indices].detach()      # shape [m, 256, 6]
        X_B = X[rest_indices]   # shape [n - m, 256, 6]

        # 转换为 batch 在前
        X_A = X_A.permute(1, 0, 2)  # [256, m, 6]
        X_B = X_B.permute(1, 0, 2)  # [256, n - m, 6]

        # 去均值
        X_A_centered = X_A - X_A.mean(dim=2, keepdim=True)
        X_B_centered = X_B - X_B.mean(dim=2, keepdim=True)

        # 分子：[256, n - m, m]
        numerator = torch.einsum('bmd,bnd->bnm', X_A_centered, X_B_centered)
        # print("numerator", numerator.shape)
        # 分母
        A_norm = torch.norm(X_A_centered, dim=2)  # [256, m]
        B_norm = torch.norm(X_B_centered, dim=2)  # [256, n - m]
        # print("A_norm", A_norm.shape)
        # print("B_norm", B_norm.shape)
        denom = torch.einsum('bm,bn->bnm', A_norm, B_norm) + 1e-8  # [256, n - m, m]
        # print("denom", denom.shape)
        corr = numerator / denom  # [256, n - m, m]

        return corr#.permute(1, 2, 0)  # [n - m, m, 256]

    @property
    def networks(self):
        base_list = [
            self.policy,
            self.qfs,
            self.target_qfs,
        ]

        return base_list

    def get_snapshot(self):
        return dict(
            policy=self.policy,
            qfs=self.qfs,
            target_qfs=self.qfs,
            log_alpha=self.log_alpha,
            policy_optim=self.policy_optimizer,
            qfs_optim=self.qfs_optimizer,
            alpha_optim=self.alpha_optimizer,
        )
