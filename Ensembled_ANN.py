def ensembled_ANN(inputs, input_len, outputs, output_len, train_len = 0, test_len, scaler):
    
    import numpy as np
    import pandas as pd
    from data_processor import series_to_supervised,evaluate
    
    # parameters
    input_len = input_len
    output_step = output_len
    num_feature = 1
    batch_size = 512
    epochs = 100
    
    test_len = test_len
    
    # split train and test(test one month)
    train_x,train_y = inputs[:-2880],outputs[:-2880]
    test_x,test_y = inputs[-2880:],outputs[-2880:]
    
    # design network
    model = Sequential()

    model.add(Dense(int(input_len/2),input_dim=input_len , activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(output_step, activation='relu'))

    adam = Adam(lr=0.001)

    model.compile(loss='mse', optimizer='adam')

    history = model.fit(train_x,train_y,
                        batch_size = batch_size,
                        epochs = epochs,
                        validation_split = 0.1,
                        callbacks=[
                            TensorBoard(log_dir='/tmp/tensorboard', write_graph=True),
                            EarlyStopping(monitor='val_loss', patience=5, mode='auto')
                        ]
                        )
    
    # make a prediction
    y_hat = model.predict(test_x)
    inv_yhat = scaler.inverse_transform(y_hat)
    
    inv_y = scaler.inverse_transform(test_y)
    
    rmse = evaluate(inv_y,inv_yhat)
    
    return inv_y, inv_yhat, rmse