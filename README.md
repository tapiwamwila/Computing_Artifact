# LSTM Streamflow Prediction

This repository contains a computing artifact focusing on using a Long Short-Term Memory (LSTM) neural network to predict daily streamflow (river discharge) in river basins. The model is trained and evaluated on the CAMELS hydrological dataset, which provides meteorological forcings and streamflow observations for numerous catchments. The goal is to explore the performance of LSTM-based models in a hydrological context, and compare the outcomes of training on a single basin versus multiple basins.

## Key Features and Goals

- **LSTM Model for Streamflow**: Implements a single-layer LSTM neural network (using PyTorch) to model rainfall-runoff processes and predict daily streamflow from meteorological inputs.
- **Single vs. Multiple Basin Training**: The project evaluates two scenarios:
  - Single Basin Model: Train and test the model on data from one specific river basin (catchment).
  - Multiple Basin Model: Train a generalized model using data from multiple basins to assess its broader applicability and performance across different catchments.
- **Hydrological Dataset (CAMELS)**: Utilizes the CAMELS dataset, which includes daily meteorological inputs (precipitation, temperature, etc.) and observed streamflow for hundreds of basins. This provides a rich, real-world dataset for training and evaluating the model.
- **Model Evaluation**: Performance is measured using metrics like Nash–Sutcliffe Efficiency (NSE) and Mean Squared Error (MSE), providing insight into predictive accuracy. NSE is a standard efficiency coefficient in hydrology (NSE = 1 is perfect, NSE > 0.75 is typically considered a very good model.

## Setup and Installation

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:  
   ```bash
   git clone https://github.com/tapiwamwila/Computing_Artifact.git
   cd Computing_Artifact
   ```

2. **Create a Python environment** (optional but recommended):  
   Use Python 3.x. You can create a virtual environment using venv or Conda. For example, with `venv` on Linux/MacOS:  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
   
3. **Install required packages**:  
   The key dependencies (listed in `scripts/packages.py`) include:  
   `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `folium`, `networkx`, `tqdm`, `numba`, `shap`, `gcsfs`, and deep learning libraries `torch` (PyTorch) and `torch_geometric`. Install these via pip:  
   ```bash
   pip install torch torch_geometric pandas numpy scikit-learn seaborn matplotlib folium shap networkx tqdm numba gcsfs
   ```  

4. **Download the CAMELS dataset**:  
   This project uses the **CAMELS (Catchment Attributes and Meteorology for Large-sample Studies)** dataset. Due to its size, the data is not included in the repo. Download the CAMELS US dataset from the [NCAR/UCAR data portal](https://ral.ucar.edu/solutions/products/camels) or the [GDEX archive](https://gdex.ucar.edu/dataset/) – specifically, you will need the time series data (`basin_timeseries_v1p2_metForcing_obsFlow.zip`). Extract the dataset on your system, which should yield a directory (for example, `basin_timeseries_v1p2_metForcing_obsFlow` containing subfolders for meteorological forcing and streamflow data).

## Dataset: CAMELS Overview

**CAMELS (Catchment Attributes and Meteorology for Large-sample Studies)** is a large-sample hydrology dataset covering hundreds of basins across the United States. It provides a wealth of data for each basin, including: 

- **Meteorological forcings**: Daily time series of precipitation, temperature (min/max), solar radiation, etc., often derived from the Daymet dataset or similar sources.
- **Streamflow observations**: Daily discharge measurements for each basin, typically provided by USGS gauges (converted to runoff in mm/day in the CAMELS data files).
- **Catchment attributes**: Static attributes of each basin (soil properties, land cover, elevation, climate indices, etc.), which can be useful for hydrologic modeling (though in this project the focus is on time series data and the LSTM model does not explicitly use these static features).

In total, CAMELS includes data for 671 basins in the contiguous United States, with records spanning multiple decades (most basins have data from the 1980s-2000s). This project uses the time series data from CAMELS to train the LSTM. Typically, the data is split into a training period and a testing (validation) period. For example, one might train on data from 1980-2000 and test on 2000-2010 for a given basin, to evaluate how well the model predicts unseen data.

## How to Train and Evaluate the Model

You can train and evaluate the LSTM model using either the provided Jupyter notebooks or the Python scripts. Below are the two approaches:

- **Using Jupyter Notebooks (recommended for exploration)**:  
  There are two notebooks provided for running experiments:
  - **Single Basin Training** (`notebooks/single_basin/single_basin_prediction.ipynb`): This notebook demonstrates training the LSTM on a single basin's data. You can open the notebook and step through the cells. It will:
    - Load and preprocess data for a specified basin (using `scripts/data_loader.py` and `scripts/data_processing.py`). You may need to set the basin ID in the notebook (for example, a USGS 8-digit basin code).
    - Initialize the LSTM model (from `scripts/model.py`) and other settings (sequence length, training epochs, learning rate, etc.).
    - Train the model on the basin's historical data. The training process will likely print the training loss per epoch and possibly validation loss.
    - Evaluate the model on a test split (or the latter portion of the time series) by predicting streamflow and comparing it to observed values.
    - Compute metrics such as NSE and MSE for the test period, and plot results (e.g. a hydrograph of observed vs predicted flow).
    - Save the trained model weights. For example, the notebook saves weights to `notebooks/single_basin/saved_model/model_weights.pth` for later use or inspection.
  - **Multiple Basins Training** (`notebooks/multiple_basins/multiple_basin_prediction.ipynb`): This notebook shows how to train the model on multiple basins at once (a "regional" model). Steps include:
    - Loading data for multiple basins (you might specify a list of basin IDs or a number of basins to use). The data loader will aggregate these into a combined dataset.
    - Training the LSTM on the pooled multi-basin data. This can demonstrate the model’s ability to learn a general representation across different catchments.
    - Evaluating the model on one or more basins (possibly including basins that were not in the training set, to test generalization).
    - Computing metrics (NSE, MSE) for the evaluation and plotting results for sample basins.

## Acknowledgements and References

- **CAMELS Dataset**: This work utilizes the CAMELS US dataset, developed by the US National Center for Atmospheric Research and collaborators. If you use this dataset, please cite the relevant papers (e.g., Newman et al., 2015 for hydrometeorological data; Addor et al., 2017 for catchment attributes). The dataset has been invaluable for advancing data-driven hydrology research.
- **Hydrological Modeling Inspiration**: The approach of applying LSTM networks to rainfall-runoff modeling is inspired by recent research in hydrology. Notably, Kratzert *et al.* (2018) demonstrated the potential of LSTMs for modeling runoff on CAMELS basins ([pangeo_lstm_example/LSTM_for_rainfall_runoff_modelling.ipynb at ...](https://github.com/kratzert/pangeo_lstm_example/blob/master/LSTM_for_rainfall_runoff_modelling.ipynb#:~:text=,for%20setting%20up%20and)), showing that deep learning models can outperform traditional hydrological models in many cases.
