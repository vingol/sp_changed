import os
import numpy as np
import pandas as pd
from keras import Sequential
from os.path import dirname, abspath
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, EarlyStopping, ReduceLROnPlateau
from data_processor import series_to_supervised,evaluate
from py_file.data_preparation import CF,sample_generator

def ANN_sp(dataset, scaler):
    # 下一步集成化，ANN的输入参数为训练集和测试集

    # parameters
    input_len = dataset[0][0].shape[1]
    output_step = 1
    batch_size = 512
    epochs = 100

    names = locals()

    for (i,[train_x,train_y,test_x,test_y]) in enumerate(dataset):

        # design network
        model = Sequential()

        model.add(Dense(int(input_len / 2), input_dim=input_len, activation='relu'))
        model.add(Dropout(0.2))

        model.add(Dense(output_step, activation='relu'))

        adam = Adam(lr=0.001)

        model.compile(loss='mse', optimizer='adam')

        history = model.fit(train_x, train_y,
                            batch_size=batch_size,
                            epochs=epochs,
                            validation_split=0.1,
                            callbacks=[
                                TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
                                EarlyStopping(monitor='val_loss', patience=5, mode='auto')
                            ]
                            )

        # make a prediction
        names['y_hat_%s' % str(i + 1)] = model.predict(test_x)
        names['inv_yhat_%s' % str(i + 1)] = scaler.inverse_transform(names['y_hat_%s' % str(i + 1)])

        test_y = pd.DataFrame(test_y)
        names['inv_y_%s' % str(i + 1)] = scaler.inverse_transform(test_y)
        names['rmse_%s' % str(i + 1)] = evaluate(names['inv_y_%s' % str(i + 1)], names['inv_yhat_%s' % str(i + 1)])

    y_hat = []
    for j in range(16):
        y_hat.append(names['inv_yhat_%s' % str(j + 1)])
    y_hat = pd.DataFrame(np.array(y_hat).reshape(16, -1).T)

    rmse = []
    for j in range(16):
        rmse.append(names['rmse_%s' % str(j + 1)])
    rmse = np.array(rmse).reshape(-1)

    return y_hat, rmse

if __name__ == '__main__':
    names = locals()
    # load power
    dir_power = dirname(dirname(abspath(__file__))) + '/power_detrended/'

    name_index_power = pd.Series(os.listdir(dir_power)).map(lambda x: x[:-4])

    for i in name_index_power:
        filename = dir_power + str(i) + '.csv'
        names['data_%s' % i] = pd.read_csv(filename, index_col=3, parse_dates=True)
        names['power_%s' % i] = pd.DataFrame(names['data_%s' % i]['power_with_trend'])
        names['power_supervised_%s' % i] = series_to_supervised(names['power_%s' % i], 48, 16)

    # load NWP
    dir_NWP = dirname(dirname(abspath(__file__))) + '/nwp_refill/'

    name_index_NWP = list(os.listdir(dir_NWP))
    name_index_NWP.remove('.DS_Store')
    name_index_NWP = pd.Series(name_index_NWP).map(lambda x: x[3:-4])

    for i in name_index_NWP:
        filename = dir_NWP + '/CN0' + str(i) + '.csv'
        names['NWP_%s' % i] = pd.read_csv(filename, index_col=0, parse_dates=True)

    index_stable, index_fluc = CF(power_66, NWP_016)
    dataset, scaler = sample_generator(power_66, index_stable, -1000, 1000)

    from pyclearsky import clearskyrad
