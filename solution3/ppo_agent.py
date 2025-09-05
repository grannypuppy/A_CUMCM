# ppo_agent.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std_init):
        super(ActorCritic, self).__init__()
        
        self.action_dim = action_dim
        # 方差是一个可训练的参数，而不是网络输出，这样更稳定
        self.action_var = torch.full((action_dim,), action_std_init * action_std_init)
        
        # Actor Network
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh() # 输出在[-1, 1]之间
        )
        
        # Critic Network
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1) # 输出一个价值
        )
        
    def set_action_std(self, new_action_std):
        self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std)

    def forward(self):
        raise NotImplementedError
        
    def act(self, state):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        return action.detach(), action_logprob.detach()
    
    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, action_std_init):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = []
        
        self.policy = ActorCritic(state_dim, action_dim, action_std_init)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, action_std_init)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def set_action_std(self, new_action_std):
        self.policy.set_action_std(new_action_std)
        self.policy_old.set_action_std(new_action_std)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action, action_logprob = self.policy_old.act(state)
        
        self.buffer.append((state, action, action_logprob))
        
        return action.flatten().numpy()

    def update(self, reward):
        # 将本次的奖励填入buffer
        # 因为是one-shot episode，所以只有一个reward
        rewards = []
        
        # 转换数据为tensor
        old_states = torch.squeeze(torch.stack([t[0] for t in self.buffer], dim=0), dim=1).detach()
        old_actions = torch.squeeze(torch.stack([t[1] for t in self.buffer], dim=0), dim=1).detach()
        old_logprobs = torch.squeeze(torch.stack([t[2] for t in self.buffer], dim=0), dim=1).detach()
        
        # 对于one-shot问题，return就是reward
        rewards.append(reward)
        rewards = torch.tensor(rewards, dtype=torch.float32)

        # 训练K个epochs
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            
            # 计算优势 Advantage
            advantages = rewards - state_values.detach()

            # 计算比率 ratio (pi_theta / pi_theta_old)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # PPO 裁剪目标函数
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            
            # 最终损失
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        
        # 更新旧策略
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空buffer
        self.buffer = []