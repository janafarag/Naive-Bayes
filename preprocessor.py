import numpy as np
import pandas as pd


class Preprocessor:

    def __init__(self):

        self._data_split_ratios = {"train": 0.8, "val": 0, "test": 0.2}  # TODO: change
        self.data_leakage_warning = False

    # Read data from file path of csv file
    def read_inflammation_data(self, file_path: str):
        inflammation_data = pd.read_csv(file_path, sep=';')  # na_values='[]'
        return inflammation_data

    def split_data(self, inflammation_data, data_split_ratios):
        # Best practice for efficiency and usability
        # Shuffle the rows of the DataFrame
        inflammation_data = inflammation_data.sample(frac=1, random_state=42)

        # Calculate the row indices for each split
        train_end = int(data_split_ratios["train"] * len(inflammation_data))
        val_end = train_end + int(data_split_ratios["val"] * len(inflammation_data))

        # Split the DataFrame into three parts with integer location, no copies
        train_data = inflammation_data.iloc[:train_end]
        val_data = inflammation_data.iloc[train_end:val_end]
        test_data = inflammation_data.iloc[val_end:]

        print(f"train data has {len(train_data)} samples")
        print(f"validation data has {len(val_data)} samples")
        print(f"test data has {len(test_data)} samples")

        return train_data, val_data, test_data

    def split_data_2(self, inflammation_data, data_split_ratios):
        # frac = 1 for entire data set, no down sampling
        # random state=42, so it is randomly generated but everytime the same when the code runs
        # for reproducibility reasons, any positive number can be use but 42 is a commonly used seed
        # because it's the answer to the ultimate question of life, the universe and everything :)
        train_data, val_data, test_data = \
            np.split(inflammation_data.sample(frac=1, random_state=42).values,
                     # train fraction of 80% of 100%, train fraction x length = index for split
                     [int(data_split_ratios["train"] * len(inflammation_data)),
                      # aufsummierter Anteil for val_data and the rest in train_data
                      int((data_split_ratios["val"] + data_split_ratios["train"]) * len(
                          inflammation_data))])

        # returns 3 arrays containing data, tuple unpacking
        print(f"train data has {len(train_data)} samples")
        print(f"validation data has {len(val_data)} samples")
        print(f"test data has {len(test_data)} samples")

        return train_data, val_data, test_data
