import torch
import torch.nn as nn

# Model Definition
class TransformerPredictor(nn.Module):
    def __init__(self, state_dim, goals_dim, mlp_grid_dim, window, num_layers, num_heads, output_dim):
        super().__init__()
        self.grid_mlp = nn.Sequential(
            nn.Linear(mlp_grid_dim[0], mlp_grid_dim[1]),  # First compression layer
            nn.ReLU(),
            nn.Linear(mlp_grid_dim[1], mlp_grid_dim[2]),  # Further reduce dimensions
            nn.ReLU(),
            nn.Linear(mlp_grid_dim[2], mlp_grid_dim[3])  # Match observation length
        )
        input_dim = state_dim*(window+1) + mlp_grid_dim[3] + goals_dim
        # self.positional_encoding = PositionalEncoding(input_dim)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, batch):
        grid_encoded = self.grid_mlp(batch['grid'])
        state = torch.flatten(batch['state'], start_dim=1, end_dim=2)
        x = torch.cat([state, grid_encoded, batch['goals']], dim=-1)
        # x = self.positional_encoding(x)
        x = self.transformer(x)
        x = self.fc(x)  # Take the last output
        return x
