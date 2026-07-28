"""PPO Agent v2 - continuous action space with Normal distribution + tanh mapping."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dims=(256, 128)):
        super().__init__()

        layers = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
            ])
            prev_dim = h_dim

        self.shared = nn.Sequential(*layers)
        self.actor_mean = nn.Linear(prev_dim, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))
        self.critic = nn.Linear(prev_dim, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, state):
        features = self.shared(state)
        mean = self.actor_mean(features)
        value = self.critic(features).squeeze(-1)
        return mean, value


class PPOAgent:
    def __init__(self, state_dim: int, action_dim: int,
                 lr=1e-3, gamma=0.99, gae_lambda=0.95,
                 clip_epsilon=0.2, entropy_coef=0.01,
                 value_coef=0.5, max_grad_norm=1.0):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"PPO Agent device: {self.device}")
        
        self.policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy = ActorCritic(state_dim, action_dim).to(self.device)
        self.old_policy.load_state_dict(self.policy.state_dict())
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=1e-5)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        self.buffer = {
            "states": [], "actions": [], "rewards": [],
            "log_probs": [], "values": [], "dones": []
        }
    
    def select_action(self, state, temperature=1.0):
        """Select continuous action and return weights + log_prob for PPO."""
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        with torch.no_grad():
            mean, value = self.policy(state_t)
            std = torch.exp(self.policy.actor_log_std) * temperature
            dist = Normal(mean, std)
            action_raw = dist.sample()
            action = torch.tanh(action_raw)  # Map to [-1, 1], then shift to [0, 1]
            weights = (action + 1) / 2  # Shift from [-1,1] to [0,1]
            weights = weights / (weights.sum() + 1e-8)  # Normalize to sum to 1
            log_prob = dist.log_prob(action_raw).sum().unsqueeze(0)
            entropy = dist.entropy().sum()

        return weights[0].cpu().numpy(), action_raw[0].cpu().numpy(), log_prob.item(), value.item(), entropy.item()
    
    def store(self, state, action, reward, log_prob, value, done):
        self.buffer["states"].append(state)
        self.buffer["actions"].append(action)  # Continuous action (raw, before tanh)
        self.buffer["rewards"].append(reward)
        self.buffer["log_probs"].append(log_prob)
        self.buffer["values"].append(value)
        self.buffer["dones"].append(done)
    
    def update(self):
        """PPO update with GAE for continuous actions."""
        if len(self.buffer["states"]) < 32:
            return None

        states = torch.FloatTensor(np.array(self.buffer["states"])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer["actions"])).to(self.device)
        old_log_probs = torch.FloatTensor(self.buffer["log_probs"]).to(self.device)
        
        rewards = self.buffer["rewards"]
        dones = self.buffer["dones"]
        
        # Compute GAE
        values_list, advantages = self._compute_gae(rewards, dones)
        returns = np.array(values_list) + np.array(advantages)
        
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)
        
        if advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # PPO epochs
        n_epochs = 8
        batch_size = min(64, len(states))
        
        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        
        for _ in range(n_epochs):
            perm = torch.randperm(len(states))
            
            for start in range(0, len(states), batch_size):
                idx = perm[start:start + batch_size]
                
                b_states = states[idx]
                b_actions = actions[idx]
                b_old_lp = old_log_probs[idx]
                b_adv = advantages_t[idx]
                b_ret = returns_t[idx]
                
                mean, value = self.policy(b_states)
                std = torch.exp(self.policy.actor_log_std)
                dist = Normal(mean, std)

                new_log_prob = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_prob - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * b_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = nn.MSELoss()(value.squeeze(), b_ret)
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_batches += 1
        
        self.old_policy.load_state_dict(self.policy.state_dict())
        self.buffer = {"states": [], "actions": [], "rewards": [],
                       "log_probs": [], "values": [], "dones": []}
        
        return (total_policy_loss / n_batches, total_value_loss / n_batches,
                total_entropy / n_batches)
    
    def _compute_gae(self, rewards, dones):
        """Compute Generalized Advantage Estimate (GAE).
        
        Terminal vs non-terminal distinction:
        - non_terminal = 1.0 - dones[i]: indicates if state i is NOT terminal (episode continues)
        - When done[i] == True, non_terminal == 0, meaning no bootstrap from future
        - At episode end (i == len(rewards)-1), even if not marked done, next_val becomes 0 via terminal value
        
        Bootstrap mechanism:
        - For non-terminal states: use next state's value estimate for bootstrapping
        - For terminal states: next_val = 0 (last_value computed from final state, but multiplied by non_terminal=0)
        - This prevents credit assignment beyond episode boundaries
        """
        # Get bootstrap value from final state
        with torch.no_grad():
            last_state = torch.FloatTensor(np.array(self.buffer["states"][-1])).unsqueeze(0).to(self.device)
            _, last_value = self.old_policy(last_state)
            last_value = last_value.item()
        
        values = []
        gae = 0
        advantages = []
        
        for i in range(len(rewards) - 1, -1, -1):
            # non_terminal = 1.0 when episode continues at state i, 0 when terminal
            non_terminal = 1.0 - dones[i]
            
            # For last timestep, use final state value; otherwise use next state's value
            if i == len(rewards) - 1:
                next_val = last_value
            else:
                next_val = self.buffer["values"][i + 1]
            
            values.append(self.buffer["values"][i])
            # GAE delta: current reward + discounted future value (if non-terminal) - current value estimate
            delta = rewards[i] + self.gamma * next_val * non_terminal - self.buffer["values"][i]
            # Accumulate GAE with decay and lambda weighting (only if non-terminal)
            gae = delta + self.gamma * self.gae_lambda * gae * non_terminal
            advantages.insert(0, gae)
        
        values.reverse()
        return values, advantages
    
    def save(self, path):
        torch.save({"policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict()}, path)
    
    def load(self, path):
        ckpt = torch.load(path, weights_only=False)
        self.policy.load_state_dict(ckpt["policy"])
        self.old_policy.load_state_dict(ckpt["policy"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
