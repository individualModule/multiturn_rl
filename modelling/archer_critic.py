"""
Borrowed and adapted from archer paper.
"""

"""
TODO -> needs to implement device switching, tensor conversion, 
etc. Completely unaligned rn, almost pseudocode

Since policy is the model from Clemgame, need to figure out how to bridge that
Need to make sure everything is on the same device.
need to make sure the shapes are all in order

"""
from typing import List, Union, Tuple

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
    
    def __init__(self, device, critic_lm, in_dim, out_dim):
        super().__init__()
        self.device = device

        # Base LM
        self.base_lm = AutoModel.from_pretrained(critic_lm).to(device)
        self.base_tokenizer = AutoTokenizer.from_pretrained(critic_lm)
        self.base_tokenizer.truncation_side = 'left'

        # Q-Critics (take state-action pairs)
        self.critic1 = CriticNetwork(in_dim, out_dim, is_q_critic=True).to(device)
        self.critic2 = CriticNetwork(in_dim, out_dim, is_q_critic=True).to(device)
        
        # V-Critics (take only states)
        self.v_critic1 = CriticNetwork(in_dim, out_dim, is_q_critic=False).to(device)
        self.v_critic2 = CriticNetwork(in_dim, out_dim, is_q_critic=False).to(device)


    def flatten_chat(self, messages: Union[list[dict], list[list[dict]]]) -> str:

        def _flatten_single(chat: List[dict]) -> str:
            return "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in chat])
        

        # If input is a single conversation (list of dicts)
        if isinstance(messages[0], dict):
            return [_flatten_single(messages)]  # wrap in list for consistency
        # If input is a batch of conversations
        elif isinstance(messages[0], list):
            return [_flatten_single(conv) for conv in messages]
        else:
            raise ValueError("Invalid input format for flatten_chat.")

    def forward(self, observation: list[list[dict]], action: list[list[dict]], detach_model=False):
        # Get embeddings
        
        flat_obs = self.flatten_chat(observation)
        flat_action = self.flatten_chat(action)

        assert len(flat_obs) == len(flat_action), "batch sizes not equal!"

        obs_ids = self.base_tokenizer(flat_obs, padding=True, return_tensors='pt', 
                                    max_length=512, truncation=True).to(self.device)
        action_ids = self.base_tokenizer(flat_action, padding=True, return_tensors='pt', 
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

    def __init__(self, policy, critic, target_critic, gamma=0.99):
        super().__init__()
        self.policy = policy
        self.critic = critic
        self.target_critic = target_critic
        self.gamma = gamma
        
    def convert_to_response_dict(self, response: str) -> dict[str, str]:

        return {
            "role": "assistant",
            "content": response
        }

    def get_policy_action(self, observations: List[List[dict]]) -> List[List[dict]]:
        """
        Sample the action from the policy using the Player class's generate_response
        return the response in the appropriate format:
                            {"role": "assistant",
                             "content": response}

        return list[list[dict]] for consistency. Same shape as actions and obs

        """
        # Generate responses using the Player
        responses = []
        for msg in observations:
            # Player.__call__ internally calls model.generate_response
            _, _, response_text = self.policy.generate_response(msg)
            response_dict = self.convert_to_response_dict(response_text)
            responses.append([response_dict])
            
        return responses


    def get_critic_values(self, observation: List[List[dict]], action: List[List[dict]]):
        """
        Get Q and V values from the critic 
        """
        # q1, q2, v1, v2 = self.critic(observation, action, detach_model=False)

        return self.critic(observation, action)

    def get_log_prob(self, observation: List[List[dict]], action: List[List[dict]]) -> torch.Tensor:
        """
        Get logprob of the generated sequence using the model's calculate_logprobs
        """
        # The underlying HuggingFace model is accessed through player.model
        return self.policy.calculate_logprobs(observation, action)

    def compute_target_q(self, observation: List[List[dict]]):
        """
        Compute target Q values using the policy generated from the sample.
        """

        with torch.no_grad():
            obs = copy.deepcopy(observation)
            pi_action = self.get_policy_action(obs)
            target_q1, target_q2, _ , _ = self.target_critic(obs, pi_action)
        
            assert target_q1.size(0) == len(observation), "Batch size mismatch"

        return target_q1, target_q2
    
    def compute_target_v(self, next_observation: List[List[dict]], action: List[List[dict]], reward: torch.Tensor, done: torch.Tensor):
        """
        Compute target V values using the  next observation (next state). 
        Action is not really used here - but needs to be passed because
        of the fwd function of critic.
        """

        with torch.no_grad():

            # Ensure reward and done are properly shaped [batch_size, 1]
            if reward.dim() == 1:
                reward = reward.unsqueeze(-1)
            if done.dim() == 1:
                done = done.unsqueeze(-1)

            done = done.float()  # Convert to float for multiplication
            _, _ , target_v1, target_v2 = self.target_critic(next_observation, copy.deepcopy(action))
            target_v1 = reward + (1 - done)*target_v1*self.gamma
            target_v2 = reward + (1 - done)*target_v2*self.gamma

        return target_v1, target_v2
    

    def soft_target_update(self, target_model, source_model, tau):
        """
        Polyak averaging: target_model = tau * source_model + (1 - tau) * target_model

        Args:
            target_model: the model we are slowly updating (e.g., EMA version)
            source_model: the model we are tracking (e.g., current LLM/actor)
            tau: interpolation factor (0 < tau <= 1), small tau = slow update
        """

        for target_param, source_param in zip(target_model.parameters(), source_model.parameters()):
            target_param.data.copy_(tau * source_param.data + (1.0 - tau) * target_param.data)


    def compute_advantages(self, rewards: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """Computes the advantage as the difference between reward and value estimate."""

        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        
        return rewards - values


class TDLoss(nn.Module):
    def __init__(self):
        super(TDLoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, q1, q2, v1, v2, target_v1, target_v2, target_q1, target_q2):

        batch_size = q1.size(0)
        assert all(x.size(0) == batch_size for x in [q2, v1, v2, target_v1, target_v2, target_q1, target_q2]), \
        "All inputs must have same batch size"

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

        def forward(self, advantage: torch.Tensor, logprobs: torch.Tensor):
            """
            REINFORCE with advantage calculation
            
            Args:
                advantage: Computed advantages from critic [batch_size, 1]
                logprobs: Log probabilities from policy [batch_size, seq_len]
            Returns:
                Combined loss from policy gradient
            """    
            
            # Ensure proper shapes
            if advantage.dim() == 1:
                advantage = advantage.unsqueeze(-1)
                
            # Compute policy gradient loss
            # Sum logprobs across sequence dimension
            pg_loss = -torch.mean(advantage * logprobs.sum(dim=1))
            
            return pg_loss