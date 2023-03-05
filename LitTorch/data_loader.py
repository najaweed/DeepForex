import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class FinDataset(Dataset):
    def __init__(self,
                 data_temporal: pd.DataFrame,
                 config: dict,

                 ):
        self.time_series = data_temporal
        self.step_predict = config['step_predict']
        self.step_share = config['step_share']
        self.freq = config['freq_obs']
        self.obs, self.target = self.x_split_observation_prediction()
    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, index):

        return self.obs[index,...], self.target[index,...]

    def x_split_observation_prediction(self):
        list_df = self.group_data(self.time_series)
        x, y = [], []
        for df in list_df:
            x.append(df.iloc[:-self.step_predict, :])
            y.append(df.iloc[-(self.step_predict + self.step_share):, :])

        x, y = np.array(x), np.array(y)
        x, y = self._normalizer(x), self._normalizer(y)
        x, y = np.expand_dims(x, axis=1), np.expand_dims(y, axis=1)
        x, y = torch.from_numpy(x), torch.from_numpy(y)
        return x.to(torch.float32), y.to(torch.float32)

    def group_data(self, x_df):
        df = x_df.copy()
        df['time'] = df.index
        df_list = []
        for group_name, df_group in df.groupby(pd.Grouper(freq=self.freq, key='time')):
            g_df = df_group.drop(columns=['time'])
            week_or_month = 20 if self.freq[1] == 'M' else 5
            if len(g_df) == int(self.freq[0]) * week_or_month:  # 5days*5weeks
                df_list.append(g_df)
        return df_list

    @staticmethod
    def _normalizer(x: np.array):
        x -= x.min(axis=2, keepdims=True)
        x /= (x.max(axis=2, keepdims=True) - x.min(axis=2, keepdims=True))
        x = 2 * x - 1
        return x


import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader


class LitFinData(pl.LightningDataModule, ):

    def __init__(self,
                 df: pd.DataFrame,
                 config: dict,
                 ):
        super().__init__()
        self.config = config
        self.df = df
        self.train_loader, self.val_loader, self.test_loader = self._gen_data_loaders()

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def test_dataloader(self):
        return self.test_loader

    def _gen_data_loaders(self):
        split_index = [int(self.df.shape[0] * self.config['split'][i] / 10) for i in range(len(self.config['split']))]
        start_index = 0
        # time_series = torch.FloatTensor(self.df[[f'{sym}.si,close' for sym in self.symbols]].values)
        data_loaders = []
        for i in range(len(self.config['split'])):
            end_index = split_index[i] + start_index
            dataset = FinDataset(self.df.iloc[start_index:end_index, :], self.config)
            data_loaders.append(DataLoader(dataset=dataset,
                                           batch_size=self.config['batch_size'],
                                           # num_workers=4,
                                           drop_last=True,
                                           # pin_memory=True,
                                           shuffle=True,
                                           ))
            start_index = end_index
        return data_loaders


# READ DATA

# df = pd.read_csv('df.csv', )
# df['time'] = pd.to_datetime(df['time'])
# df.set_index('time', inplace=True)
# # df.drop('time', axis=1, inplace=True)
# df = df.dropna(axis=0, how='any')
# df = df.dropna(axis=0, how='any')
# #
# import matplotlib.pyplot as plt
# plt.figure(11)
# plt.plot(df['USD'])
# plt.plot(df.ewm(alpha=0.005).mean()['USD'])
#
#
# plt.show()

# config = {
#     'batch_size': 1,
#     'num_nodes': 6,
#     'freq_obs': '3W',
#     'step_predict': 3,
#     'step_share': 0,
#     'split': (7, 2, 1),
#     'in_channels': 2 * 3 * 4,
#     'hidden_channels': 256,
#     'out_channels': 2 * 3,
#     'num_layers': 8,
#     'learning_rate': 5e-3,
#     'in_features': 12 * 6,
#     'hidden_features': 128,
#     'out_features': 3 * 6,
# }
# lit_data = LitFinData(df, config)
#
# lit_val = lit_data.val_loader
# print(len(lit_data.train_dataloader()))
# for a, b in lit_data.train_dataloader():
#     print(a.shape, b.shape)
#     break
#     #print(a)
