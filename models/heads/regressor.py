import torch
import torch.nn as nn


class PredictorRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PredictorRegressor, self).__init__()