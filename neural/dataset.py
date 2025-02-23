import torch
import numpy as np
from torch.utils.data import Dataset


# Custom Dataset
class TargetDataset(Dataset):
    def __init__(self, df, window):
        self.df = df
        self.window = window
        self.index = df.loc[df["step"]>window, ["step"]].reset_index(drop=False)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):

        # Index the data
        step_idx = self.index.loc[idx, 'index']
        step_row = self.df.loc[step_idx]

        join_state = lambda row: [row["target_x"], row["target_y"]] + row["direction"].tolist()
        sequence_data = self.df.loc[step_idx-self.window:step_idx]
        state = sequence_data.apply(join_state, axis=1).tolist()
        grid = step_row['grid_encoding']
        goals = step_row['goals_encoding']
        action = step_row['action_encoding']

        #"There should be only one instance end in the sequence"
        assert sequence_data['instance_end'].values[0:4].sum() == 0 

        if step_row['instance_end']==1:
            next_state = np.zeros(7)
            next_state[6] = 1
        else:
            next_row = self.df.loc[step_idx+1]
            next_state = np.array(join_state(next_row) + [0])

        item = {
            "state": torch.tensor(state, dtype=torch.float32),
            "grid": torch.tensor(grid, dtype=torch.float32),
            "goals": torch.tensor(goals, dtype=torch.float32),
            "action": torch.tensor(action, dtype=torch.float32),
            "next_state": torch.tensor(next_state, dtype=torch.float32)
        }

        return item