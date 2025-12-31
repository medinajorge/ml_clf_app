# %%
#Packages
import numpy as np
import pandas as pd

# %% [md]
"""
# Tests
"""
# %%
%run main_gui.py
# %%
df = pd.read_csv('data/dataset_OOD_hq_classified.csv')
df['abstained'].mean() * 100
# %%
# TODO: create executable and test it
# TODO: add documentation
# %%
