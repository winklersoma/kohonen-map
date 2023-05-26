import os

import pandas as pd


class DataLoader:
    def __init__(self, file_name: str):
        # Data folder
        data_folder = "../data"
        # Create data folder, if it does not exist
        os.makedirs(data_folder, exist_ok=True)
        # Create file path
        file = os.path.join(data_folder, file_name)
        self.data = pd.read_csv(file)
        # Save a non-numerical column before scaling
        species = pd.Categorical(self.data['Species']).codes
        self.data = self.data.drop(columns=['Id', 'Species'], axis=1)
        # Implement MinMax scaling
        self.data = (self.data - self.data.min()) / (self.data.max() - self.data.min())
        # Put back the missing column
        self.data['Species'] = species
