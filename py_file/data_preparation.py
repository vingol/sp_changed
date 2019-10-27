import os
import datetime
import numpy as np
import pandas as pd
from os.path import dirname, abspath
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from data_processor import series_to_supervised,evaluate

names = locals()
# load power
dir_power = dirname(dirname(abspath(__file__))) + '/power_detrended/'

name_index_power = pd.Series(os.listdir(dir_power)).map(lambda x:x[:-4])

for i in name_index_power:
    filename = dir_power+str(i)+'.csv'
    names['data_%s' % i] = pd.read_csv(filename, index_col=3, parse_dates=True)
    names['power_%s' % i] = pd.DataFrame(names['data_%s' % i]['power_with_trend'])
    names['power_supervised_%s' % i] = series_to_supervised(names['power_%s' % i], 48, 16)

#load NWP
dir_NWP = dirname(dirname(abspath(__file__))) + '/nwp_refill/'

name_index_NWP = list(os.listdir(dir_NWP))
name_index_NWP.remove('.DS_Store')
name_index_NWP = pd.Series(name_index_NWP).map(lambda x:x[3:-4])

for i in name_index_NWP:
    filename = dir_NWP + '/CN0'+str(i)+'.csv'
    names['NWP_%s' % i] = pd.read_csv(filename, index_col=0, parse_dates=True)

def CF(power, NWP):
    cloud = pd.DataFrame(NWP.cloud_amount)
    cloud['cloud_diff'] = cloud.diff(1)
    cloud = cloud.fillna(0)

    cloud['fluctuate'] = [0 if ((cloud.cloud_amount[i]>95 or cloud.cloud_amount[i]<5) and abs(cloud.cloud_diff[i])<1)
                          else 1
                          for i in range(len(cloud))]

    cloud = cloud.loc[power.index]

    index_stable = [x for x in cloud[cloud.fluctuate == 0].index if x < datetime.datetime(2019,1,1)]
    index_fluc = [x for x in cloud[cloud.fluctuate == 1].index if x < datetime.datetime(2019,1,1)]

    power_stable = power.loc[index_stable]
    power_fluc = power.loc[index_fluc]

    return index_stable,index_fluc



def sample_generator(power, index_stable, train_len, test_len):
    # 返回归一化后的两个分类下的输入输出
    train_len = train_len
    test_len = test_len

    # normalizaion
    index = power.index
    scaler = MinMaxScaler(feature_range=(0, 1))
    power_normalized = pd.DataFrame(scaler.fit_transform(power))
    power_normalized['index'] = index
    power_normalized = power_normalized.set_index('index')

    power_supervised = series_to_supervised(power_normalized, 48, 16)

    index_of_inputs = list(range(16)) + list(range(0, -8, -1))
    inputs = power_supervised.iloc[:, :48]
    outputs = power_supervised.iloc[:, 48:]

    for i in range(16):
        names['output_%s' % str(i + 1)] = outputs.iloc[:, i]

    outputset = []
    for i in range(16):
        names['output_stable_%s' % str(i + 1)] = names['output_%s' % str(i + 1)].loc[
            list(map(lambda x: x + datetime.timedelta(minutes=15 * i), index_stable))]
        names['output_stable_%s' % str(i + 1)] = names['output_stable_%s' % str(i + 1)].dropna()
        outputset.append(names['output_stable_%s' % str(i + 1)])

    inputset = []
    for i in range(16):
        names['input_stable_%s' % str(i + 1)] = inputs.loc[names['output_stable_%s' % str(i + 1)].index]
        inputset.append(names['input_stable_%s' % str(i + 1)])

    dataset = []
    for i in range(len(inputset)):
        # train_x_set,train_y_set,test_x_set,test_y_set
        dataset.append([inputset[i][:train_len], outputset[i][:train_len], inputset[i][train_len + test_len:], outputset[i][train_len + test_len:]])

    return dataset, scaler

if __name__ == '__main__':
    index_stable, index_fluc = CF(power_66, NWP_016)
    dataset, scaler = sample_generator(power_66, index_stable, -1000, 1000)
