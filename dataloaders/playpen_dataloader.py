"""
Implement a dataloader to load batches 

"""

from torch.utils.data import Dataset
import torch

class StepRolloutDataloader(Dataset):
    """
    Goal of this fn is to enable sampling from the buffer, and also to enable random sampling
    individual observations without the entire trajectory. We want to break linearity in training 
    and stop models from overfitting on sequences.
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