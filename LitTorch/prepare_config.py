import pandas as pd
from LitTorch.data_loader import LitFinData
import torch
import torch.nn as nn

# READ DATA
df = pd.read_csv('df.csv', )
df['time'] = pd.to_datetime(df['time'])
df.set_index('time', inplace=True)
# df.drop('time', axis=1, inplace=True)
df = df.dropna(axis=0, how='any')
df = df.diff().dropna(axis=0, how='any').ewm(alpha=0.005).mean()
# READ DATA
config_data_loader = {
    # config dataset and dataloader
    'batch_size': 5,
    'freq_obs': '3W',
    'step_predict': 1,
    'step_share': 1,
    'split': (7, 2, 1),
}
# FIND MODEL CONFIG BASED ON DATALOADER
lit_data = LitFinData(df, config_data_loader)
lit_val = lit_data.val_loader
in_shape, out_shape = None, None


def calculate_kernel_last_layer(sample_input, shape_last_layer=(32, 32)):
    'messy messy messy'
    kernel = [0, 0]
    shape_input = sample_input.size()

    for h in range(1, 100):
        # print('H', h)
        model = nn.Conv2d(1,
                          1,
                          kernel_size=(shape_input[-2], h),
                          bias=False)
        # print(shape_input[-i])
        if shape_last_layer[-1] < shape_input[-1]:
            # calculate kernel
            x = model(sample_input)
            if x.shape[-1] == shape_last_layer[-1]:
                print(x.shape)

                kernel[-1] = h
                break
        elif shape_last_layer[-1] == shape_input[-1]:
            kernel[-1] = 1

        for w in range(1, 100):
            # print('W', w)
            model = nn.Conv2d(1,
                              1,
                              kernel_size=(w, shape_input[-1]),
                              bias=False)
            # print(shape_input[-i])
            if shape_last_layer[-2] < shape_input[-2]:
                # calculate kernel
                x = model(sample_input)
                if x.shape[-2] == shape_last_layer[-2]:
                    # print(x.shape)

                    kernel[-2] = w
                    break
            elif shape_last_layer[-2] == shape_input[-2]:
                kernel[-2] = 1

    return kernel


# find kernel of deepNet to last layer
kernel_last_layer = None
for a, b in lit_data.train_dataloader():
    in_shape, out_shape = a.size(), b.size()
    print('input_shape ', in_shape, '|', 'output_shape ', out_shape)
    kernel = calculate_kernel_last_layer(a, out_shape)
    kernel_last_layer = kernel
    break
print('find last kernel = ', kernel_last_layer)
config_conv = {
    # config network
    # config conv layers
    'in_channels': 1,
    'hidden_channels': 64,
    'out_channels': 1,
    # 'kernel_first': [4, 11],
    'kernel_last': kernel_last_layer,
    'num_res_layers': 8,
}

config = config_data_loader | config_conv # == {**config_data_loader , **config_conv}
