#!/usr/bin/env python
import warnings
from typing import Tuple
import numpy as np
import torch
import tqdm
warnings.filterwarnings("ignore")

# Define the device for torch computations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_epoch(model, optimizer, loader, loss_func, epoch):
    """Train model for a single epoch.

    :param model: A torch.nn.Module implementing the LSTM model
    :param optimizer: One of PyTorch's optimizer classes.
    :param loader: A PyTorch DataLoader, providing the training
        data in mini batches.
    :param loss_func: The loss function to minimize.
    :param epoch: The current epoch (int) used for the progress bar
    """
    model.train()
    pbar = tqdm.tqdm(loader)
    pbar.set_description(f"Epoch {epoch}")
    for xs, ys in pbar:
        optimizer.zero_grad()
        xs, ys = xs.to(DEVICE), ys.to(DEVICE)
        y_hat = model(xs)
        loss = loss_func(y_hat, ys)
        loss.backward()
        optimizer.step()
        pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

def eval_model(model, loader) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the model.

    :param model: A torch.nn.Module implementing the LSTM model
    :param loader: A PyTorch DataLoader, providing the data.
    :return: Two torch Tensors, containing the observations and 
        model predictions
    """
    model.eval()
    obs = []
    preds = []
    with torch.no_grad():
        for xs, ys in loader:
            xs = xs.to(DEVICE)
            y_hat = model(xs)
            obs.append(ys)
            preds.append(y_hat)
    return torch.cat(obs), torch.cat(preds)
        
def calc_nse(obs: np.array, sim: np.array) -> float:
    """Calculate Nash-Sutcliff-Efficiency.

    :param obs: Array containing the observations
    :param sim: Array containing the simulations
    :return: NSE value.
    """
    sim = np.delete(sim, np.argwhere(obs < 0), axis=0)
    obs = np.delete(obs, np.argwhere(obs < 0), axis=0)

    sim = np.delete(sim, np.argwhere(np.isnan(obs)), axis=0)
    obs = np.delete(obs, np.argwhere(np.isnan(obs)), axis=0)

    denominator = np.sum((obs - np.mean(obs)) ** 2)
    numerator = np.sum((sim - obs) ** 2)
    nse_val = 1 - numerator / denominator

    return nse_val
