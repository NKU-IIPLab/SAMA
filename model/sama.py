# coding: utf-8
import torch
import torch.nn as nn


class SAMA(nn.Module):
    """
    the sama model, including four parts: job description encoder, skill prediciton, skill
    refinement, job requirement generation.
    """
    def __init__(self, dc, config):
        super(SAMA, self).__init__()
        self.device = config.device
