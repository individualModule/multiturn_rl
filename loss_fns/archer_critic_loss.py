import torch
import torch.nn as nn
import numpy as np
from typing import Tuple


class TDLoss(nn.Module):
    def __init__(self, critic, target_critic, policy):
        super(TDLoss, self).__init__()
        self.criterion = nn.MSELoss()



    def forward(self, observation, action, reward, next_observation):

        q1, q2, v1, v2 = self.ciritc()


        pass