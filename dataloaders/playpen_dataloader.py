"""
Implement a dataloader to load batches 

"""

from torch.utils.data import Dataset
import torch

class StepRolloutDataloader(Dataset):
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
                    'reward': current['info'].get('reward', 0),
                    'next_obs': next_step['context'],
                    'done': current['done']
                })
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return sample