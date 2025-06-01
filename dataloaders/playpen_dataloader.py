"""
Implement a dataloader to load batches 

"""

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch


class StepRolloutDataset(Dataset):
    """
    Goal of this fn is to enable sampling from the buffer, and also to enable random sampling
    individual observations without the entire trajectory. We want to break linearity in training 
    and stop models from overfitting on sequences.
    """
    def __init__(self, trajectories: list[dict[str, dict]]):
        self.samples = []
        # Flatten trajectories into individual transition samples

        # context returns a list of previous interactions (each interaction is a dict)}
        # we don't return None for next obs because it will break the code.
        # the fn for target q will just use the reward instead of calc. q value since we send info that episode is done.
        for trajectory in trajectories:
            for i in range(len(trajectory)):  
                current = trajectory[i]
                next_step = trajectory[i + 1] if i+1 < len(trajectory) else None
                self.samples.append({
                    'obs': current['context'],
                    'action': current['response'],
                    'reward': torch.tensor(
                        current['info']['response_score'] if next_step else current['info']['episode_score'], 
                        dtype=torch.float
                    ),
                    'next_obs': next_step['context'] if next_step else current['context'],  
                    'done': torch.tensor(current['done'], dtype=torch.bool)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample
    
    
class StepRolloutDatasetV3(Dataset):
    """
    Goal of this fn is to enable sampling from the buffer, and also to enable random sampling
    individual observations without the entire trajectory. We want to break linearity in training 
    and stop models from overfitting on sequences.
    

    Compatible with V3 of clemcore
    """
    def __init__(self, trajectories):
        self.samples = []
        # Flatten trajectories into individual transition samples
        for trajectory in trajectories:
            for i in range(len(trajectory) - 1):  # -1 to handle next_obs
                current = trajectory[i]
                next_step = trajectory[i + 1]

                self.samples.append({
                    'obs': current['context'],
                    'action': current['response'],
                    'reward': torch.tensor(current['turn_score'], dtype=torch.float),
                    'next_obs': next_step['context'],
                    'done': torch.tensor(current['done'], dtype=torch.bool)
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample


def custom_collate_fn(batch):
    """
    Custom collate function to handle batching of samples.
    """
    obs: list[list[dict[str, str]]] = [item['obs'] for item in batch]
    actions: list[list[dict[str, str]]] = [item['action'] for item in batch]
    rewards: list[int] = torch.stack([item['reward'] for item in batch])
    next_obs: list[dict[str, str]] = [item['next_obs'] for item in batch]
    dones = torch.stack([item['done'] for item in batch])
    
    return {
        'obs': obs,
        'action': actions,
        'reward': rewards,
        'next_obs': next_obs,
        'done': dones
    }