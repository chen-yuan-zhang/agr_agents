import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import hydra
import numpy as np
import pandas as pd
from tqdm import tqdm
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from .model import TransformerPredictor
from .logger import Logger, WanDBLogger
from .dataset import TargetDataset, preprocess_data


loggers = {
    "base": Logger,
    "wandb": WanDBLogger,
}

metric_funcs = {
    "f1": lambda y_true, y_pred: f1_score(y_true, y_pred, average="macro"),
    "accuracy": accuracy_score,
    "precision": lambda y_true, y_pred: precision_score(y_true, y_pred, average="macro"), 
    "recall": lambda y_true, y_pred: recall_score(y_true, y_pred, average="macro"), 
}

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg : DictConfig) -> None: 

    SIZE = cfg["env"]["size"]
    OBSERVATION_WINDOW = cfg["model"]["observation_window"]
    NEPOCHS = cfg["model"]["nepochs"]
    LR = cfg["model"]["lr"]
    NUM_LAYERS = cfg["model"]["num_layers"]
    NUM_HEADS = cfg["model"]["num_heads"]
    ENABLE_HIDDEN_COST = int(cfg["env"]["enable_hidden_cost"])
    save_path = cfg["model"]["save_path"]
    keys = ["epoch", "loss_train", "loss_valid"]

    # Add additional parameters config:
    parameters = OmegaConf.to_container(cfg, resolve=True)
    parameters["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the logger
    metrics = ["partition"] + list(metric_funcs.keys())
    logger = loggers[cfg.logger.type](
        parameters, keys, metrics
    )

    # Load the data
    data_runs_path = cfg["data"]["data_runs_path"].format(SIZE, ENABLE_HIDDEN_COST)
    data_runs = pd.read_csv(data_runs_path)
    data_runs = data_runs[~data_runs['action'].isna()]
    data_scenarios_path = cfg["data"]["data_scenarios_path"].format(SIZE, ENABLE_HIDDEN_COST)
    data_scenarios = pd.read_csv(data_scenarios_path)
    data = pd.merge(data_runs, data_scenarios[["layout", "scenario", "goals"]], on=["layout", "scenario"])

    # Preprocess the data
    data = preprocess_data(data, OBSERVATION_WINDOW, SIZE)

    nlayouts = len(data["layout"].unique())
    data.loc[data["layout"]<=nlayouts*0.7, "PARTITION"] = "TRAIN"
    data.loc[(data["layout"]>nlayouts*0.7) & (data["layout"]<=nlayouts*0.85), "PARTITION"] = "VALID"
    data.loc[data["layout"]>nlayouts*0.85, "PARTITION"] = "TEST"
    
    train_data = data[data["PARTITION"]=="TRAIN"]
    valid_data = data[data["PARTITION"]=="VALID"]
    test_data = data[data["PARTITION"]=="TEST"]

    # Datasets
    train_dataset = TargetDataset(train_data, OBSERVATION_WINDOW)
    valid_dataset = TargetDataset(valid_data, OBSERVATION_WINDOW)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=False)

    # Dynamically determine input dimension
    state_dim = 6
    action_dim = len(data.iloc[0]['action_encoding'])
    goals_dim = len(data.iloc[0]['goals_encoding'])
    grid_dim = data["grid_encoding"][0].shape[0]
    mlp_grid_dim = [grid_dim, 256, 64, 30]

    goals_dim = len(data.iloc[0]['goals_encoding'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TransformerPredictor(state_dim=state_dim, goals_dim=goals_dim, mlp_grid_dim=mlp_grid_dim, 
                                 window=OBSERVATION_WINDOW, num_layers=NUM_LAYERS, num_heads=NUM_HEADS, 
                                 output_dim=action_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # Training Loop
    for epoch in range(NEPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", leave=False)
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch["action"])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item(), avg_loss=total_loss / (progress_bar.n + 1))

        model.eval()
        val_loss = 0
        with torch.no_grad():
            progress_bar = tqdm(valid_dataloader, desc=f"Epoch {epoch+1}", leave=False)
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(batch)
                loss = criterion(outputs, batch["action"]).item() 
                val_loss += loss 
                progress_bar.set_postfix(loss=val_loss, avg_loss=val_loss / (progress_bar.n + 1))
        
        epoch_log = {"epoch": epoch+1, "loss_train": total_loss/len(train_dataloader), "loss_valid": val_loss/len(valid_dataloader)}
        logger.log_epoch(epoch_log)

    # Save the model
    torch.save(model.state_dict(), f"{save_path}/{logger.run_name}.pth")

    # Evaluate the model  
    dataset = TargetDataset(data, OBSERVATION_WINDOW)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    model.eval()
    predictions = []
    for batch in tqdm(dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = model(batch)#.tolist()
        pred = torch.nn.functional.softmax(pred, dim=1).detach().cpu().tolist()
        predictions += pred 

    dataset.index["probs"] = predictions
    data_pred = pd.merge(data.reset_index(drop=False), dataset.index, on=["index"])
    data_pred["pred"] = data_pred["probs"].apply(lambda x: np.argmax(x))

    for p in ["TRAIN", "VALID", "TEST"]:
        data_pred_p = data_pred[data_pred["PARTITION"] == p]
        y_pred = data_pred_p["pred"].values
        y_true = data_pred_p["action"].astype(int).values

        metrics = {"partition": p}
        for metric in metric_funcs.keys():
            metrics[metric] = metric_funcs[metric](y_true, y_pred)
        logger.log_metrics(metrics)


    logger.close()
    print("FINISHED RUNNING")

if __name__ == "__main__":
    main()