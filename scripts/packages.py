#the necessary libraries
from pathlib import Path
from typing import Tuple
import warnings
from sklearn.metrics import mean_squared_error
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import warnings
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
from numba import njit
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import tqdm
import gcsfs
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import folium
from IPython.display import display, IFrame
from torch_geometric.nn import GCNConv
import os
import glob
import pandas as pd
from typing import List, Dict
import torch_geometric
from torch_geometric.data import Data
import torch
import torch_geometric
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.optim as optim
import networkx as nx
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
import shap