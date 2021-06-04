#%%
import numpy as np
import tensorflow as tf
import keras
from keras.regularizers import l2,l1
import pandas as pd 
from keras.utils import to_categorical, normalize
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sklearn.preprocessing as prep

data = pd.read_csv('ETTh1.csv')
data = pd.DataFrame(data,columns = ['HUFL'])
data = data.to_numpy()
n_inp=24
n_out=1
#%%
def data_split_training(data, n_inp = 24, n_out = 1, test_split = None, validation_split = None):
    #Data pre-processing for training
    y = [data[j+n_inp:j+n_inp+n_out] for j in range(0,len(data)-(n_inp+n_out))]
    #y = normalize(np.array(y), axis=0, order =2)
    y = np.array(y)
    #y = prep.scale(y[:,:,0])
    x = [data[i:i+n_inp] for i in range(0, len(data)-(n_inp+n_out))]
    #x = normalize(np.array(x), axis=0, order =2)
    x = np.array(x)
    x = prep.scale(x[:,:,0])

    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=test_split ,shuffle=False)

    x_train = np.reshape(x_train, (len(x_train), -1))
    x_test = np.reshape(x_test, (len(x_test), -1))

    y_train = np.reshape(y_train, (len(y_train), -1))
    y_test = np.reshape(y_test, (len(y_test), -1))

    #Validation Split
    x_train, x_val, y_train, y_val = train_test_split(x_train,y_train, test_size=validation_split, shuffle=False)
    
    return x_train, x_test, x_val, y_train, y_test, y_val


x_train, x_test, x_val, y_train, y_test, y_val = data_split_training(data, n_inp = 24, n_out = 1, test_split = 0.2, validation_split = 0.2)

#%%
model = keras.Sequential()
activation = model.add(keras.layers.LeakyReLU())
model.add(keras.layers.Dense(n_inp, input_shape=(n_inp,), kernel_regularizer=l2(0.01), activation='relu'))
model.add(keras.layers.Dense(75, kernel_regularizer=l2(0.005), activation='relu'))
model.add(keras.layers.Dense(150, kernel_regularizer=l2(0.0008), activation='relu'))
model.add(keras.layers.Dense(250, kernel_regularizer=l2(0.0001), activation='relu'))
model.add(keras.layers.Dense(100, kernel_regularizer=l2(0.0002), activation='relu'))
model.add(keras.layers.Dense(n_out, activation=activation))
optimizer = keras.optimizers.Adam(learning_rate=0.00002)
model.compile(optimizer=optimizer, loss='mse')
#%%
model.fit(x=x_train, y=y_train, validation_data = (x_val,y_val),shuffle=False, epochs=60, verbose=1)

y_predict = model.predict(x_test)

#Heat map

for i in range(0,6):
    a = model.get_weights()[i*2]
    plt.imshow(a, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.show()

#%%
#MSE 
MSE = keras.losses.MeanSquaredError(reduction="auto", name="mean_squared_error")
print(MSE(y_predict,y_test))
#%%
#test display

plt.plot(y_predict[24*10:24*19,-1], label = 'predictions')
plt.plot(y_test[24*10:24*19,-1], label = 'test')
plt.legend()
