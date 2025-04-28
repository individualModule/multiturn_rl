"""
Borrowed and adapted from archer paper.
"""

"""
TODO -> needs to implement device switching, tensor conversion, 
etc. Completely unaligned rn, almost pseudocode
"""

import torch
import transformers
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Tuple
import torch.nn as nn
import numpy as np
import copy

class CriticNetwork(nn.Module):
    """Base critic network architecture
    if Q is calculated, in dim is *2.

    Returns the output of the Q and V function
    """
    def __init__(self, in_dim: int, out_dim: int, is_q_critic: bool = True):
        super().__init__()
        self.in_features = in_dim * 2 if is_q_critic else in_dim
        
        # critic network
        self.net = nn.Sequential(
            nn.Linear(self.in_features, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, in_dim),
            nn.ReLU(),
            nn.Linear(in_dim, out_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class DoubleCritic(nn.Module):
    
    def __init__(self, device, accelerator, critic_lm, cache_dir, in_dim, out_dim):
        super().__init__()
        self.device = device
        self.accelerator = accelerator

        # Base LM
        self.base_lm = AutoModel.from_pretrained(critic_lm, cache_dir=cache_dir).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm, cache_dir=cache_dir)
        self.base_tokenizer.truncation_side = 'left'
        
        # Q-Critics (take state-action pairs)
        self.critic1 = CriticNetwork(in_dim, out_dim, is_q_critic=True).to(device)
        self.critic2 = CriticNetwork(in_dim, out_dim, is_q_critic=True).to(device)
        
        # V-Critics (take only states)
        self.v_critic1 = CriticNetwork(in_dim, out_dim, is_q_critic=False).to(device)
        self.v_critic2 = CriticNetwork(in_dim, out_dim, is_q_critic=False).to(device)

    def forward(self, observation, action, detach_model=False):
        # Get embeddings
        obs_ids = self.base_tokenizer(observation, padding=True, return_tensors='pt', 
                                    max_length=512, truncation=True).to(self.device)
        action_ids = self.base_tokenizer(action, padding=True, return_tensors='pt', 
                                       max_length=512, truncation=True).to(self.device)
        
        # Get state and action representations
        if detach_model:
            with torch.no_grad():
                lm_states = self.base_lm(**obs_ids).pooler_output
                action_states = self.base_lm(**action_ids).pooler_output
        else:
            lm_states = self.base_lm(**obs_ids).pooler_output
            action_states = self.base_lm(**action_ids).pooler_output
            
        # Concatenate for Q-critics
        q_states = torch.cat([lm_states, action_states], dim=1)
        
        return (
            self.critic1(q_states),
            self.critic2(q_states),
            self.v_critic1(lm_states),
            self.v_critic2(lm_states)
        )
    

class ArcherAgent(nn.Module):
    """
    A class wrapping ArCHer into an abstract agent. 
    Contains the fns to interact with the policy, critics, obtain targets, etc.
    """

    def __init__(self, policy, critic, target_critic):
        super().__init__()
        self.policy = policy
        self.critic = critic
        self.target_critic = target_critic
        pass

    def get_policy_action(self, observation):
        """
        Sample the action from the policy
        Take in observation, generate response.
        Return -> response
        """

        raise NotImplementedError

    def calculate_targets(self, critic, action, observation, reward, done, next_observation):
        target_q1, target_q2, _, _ = self.critic(copy.deepcopy(observation), action)
        pass


    def get_critic_values(self):
        """
        Get Q and V values from the critic 
        """
        # q1, q2, v1, v2 = self.critic(observation, action, detach_model=False)
        raise NotImplementedError

    def get_log_prob(self, observation, action):
        """
        Get logprob of the generated sequence
        """
        raise NotImplementedError
    

    def compute_target_q(self, observation):
        """
        Compute target Q values using the policy generated from the sample.
        """
        obs = copy.deepcopy(observation)
        pi_action = self.get_policy_action(obs)
        target_q1, target_q2, _ , _ = self.target_critic(obs, pi_action)

        return target_q1, target_q2
    
    def compute_target_v(self, next_observation, action, reward, done):
        """
        Compute target V values using the  next observation (next state). 
        Action is not really used here - but needs to be passed because
        of the fwd function of critic.
        """

        _, _ , target_v1, target_v2 = self.target_critic(next_observation, copy.deepcopy(action))
        target_v1 = reward + (1 - done)*target_v1.flatten()*self.gamma
        target_v2 = reward + (1 - done)*target_v2.flatten()*self.gamma

        return target_v1, target_v2
    

    
    def soft_target_update(self, target, param):
        """
        update target Q and V using Polyak averaging;
        """
        raise NotImplementedError

    
    def compute_advantages(rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Computes the advantage as the difference between reward and value estimate."""
        return rewards - values


class TDLoss(nn.Module):
    def __init__(self, critic, target_critic, policy):
        super(TDLoss, self).__init__()
        self.criterion = nn.MSELoss()



    def forward(self, q1, q2, v1, v2, target_v1, target_v2, target_q1, target_q2):

        q1_loss = self.criterion(q1, target_v1)
        q2_loss = self.criterion(q2, target_v2)
        v1_loss = self.criterion(v1, target_q1)
        v2_loss = self.criterion(v2, target_q2)

        loss = q1_loss + q2_loss + v1_loss + v2_loss
        return loss
        
        

class Reinforce(nn.Module):
        def __init__(self):
            super(Reinforce, self).__init__()
            self.criterion = nn.MSELoss()

        def forward(self, advantage, log_prob):
            """
            REINFORCE WITH BASELINE
            values -> logprob values used to calculate (value) advantage Loss
            """    
            values, log_prob, mask = log_prob
            advantage_loss = self.criterion((advantage * mask), (values * mask))

            with torch.no_grad():
                residual_advantage = advantage - values

            pg_loss = -torch.mean(torch.sum(residual_advantage * log_prob * mask, dim=1))



            return advantage_loss, pg_loss, residual_advantage
