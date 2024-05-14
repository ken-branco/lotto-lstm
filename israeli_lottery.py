import numpy as np
import pandas as pd

from keras.models import Sequential, model_from_json
from keras.layers import LSTM, Dense, Bidirectional, Dropout

from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler


# df = pd.read_csv("IsraeliLottery_1012.csv")
df = pd.read_csv("draw_results.csv")

df.head()

df.tail()

df.info()

df.describe()

# df.drop(['Game', 'Date'], axis=1, inplace=True)
df.drop(['date', 'when'], axis=1, inplace=True)

df.head()

scaler = StandardScaler().fit(df.values)
transformed_dataset = scaler.transform(df.values)
transformed_df = pd.DataFrame(data=transformed_dataset, index=df.index)

transformed_df.head()

# All our games
number_of_rows = df.values.shape[0]
number_of_rows

# Amount of games we need to take into consideration for prediction
window_length = 7
window_length 

# Balls counts
number_of_features = df.values.shape[1]
number_of_features

X = np.empty([ number_of_rows - window_length, window_length, number_of_features], dtype=float)
print(X)

y = np.empty([ number_of_rows - window_length, number_of_features], dtype=float)
print(y)

for i in range(0, number_of_rows-window_length):
    X[i] = transformed_df.iloc[i : i+window_length, 0 : number_of_features]
    y[i] = transformed_df.iloc[i+window_length : i+window_length+1, 0 : number_of_features]

X.shape
y.shape

print(X[0])
print(y[0])

print(X[1])
print(y[1])

# Recurrent Neural Netowrk (RNN) with Long Short Term Memory (LSTM)
# Importing the Keras libraries and packages
batch_size = 100

# Initialising the RNN
model = Sequential()

# Load model architecture
with open('my_lstm_model.json', 'r') as f:
    model_json = f.read()
model = model_from_json(model_json)

# # Load model weights
model.load_weights('my_lstm_model.weights.h5')


# Adding the input layer and the LSTM layer
# model.add(Bidirectional(LSTM(240,
#                         input_shape = (window_length, number_of_features),
#                         return_sequences = True)))
# Adding a first Dropout layer
# model.add(Dropout(0.2))
# Adding a second LSTM layer
# model.add(Bidirectional(LSTM(240,
#                         input_shape = (window_length, number_of_features),
#                         return_sequences = True)))
# Adding a second Dropout layer
# model.add(Dropout(0.2))
# Adding a third LSTM layer
# model.add(Bidirectional(LSTM(240,
#                         input_shape = (window_length, number_of_features),
#                        return_sequences = True)))
# # Adding a fourth LSTM layer
# model.add(Bidirectional(LSTM(240,
#                         input_shape = (window_length, number_of_features),
#                         return_sequences = False)))
# Adding a fourth Dropout layer
# model.add(Dropout(0.2))
# Adding the first output layer
# model.add(Dense(59))
# Adding the last output layer
# model.add(Dense(number_of_features))

model.compile(optimizer=Adam(learning_rate=0.0001), loss ='mse', metrics=['accuracy'])
# model.fit(x=X, y=y, batch_size=100, epochs=500, verbose=2)
# model.fit(x=X, y=y, batch_size=100, epochs=1100, initial_epoch=1000, verbose=2)

# Save model architecture
# with open('my_lstm_model.json', 'w') as f:
#    f.write(model.to_json())

# Save model weights
# model.save_weights('my_lstm_model.weights.h5', overwrite=True)

# winners = []

# for i in range(1, len(df) - 5):  # adjust the range to avoid indexing errors
#     to_predict = df.iloc[-7 - i:-i]
#     to_predict = np.array(to_predict)
#     scaled_to_predict = scaler.transform(to_predict)
#     y_pred = model.predict(np.array([scaled_to_predict]))
#     prediction = scaler.inverse_transform(y_pred).astype(int)[0]
#     print("The predicted numbers in the last lottery game are:", prediction)
#     actual = df.iloc[-i]
#     actual = np.array(actual)
#     print("The actual numbers in the last lottery game were:", actual)
#     if np.array_equal(prediction, actual):
#         winners.append(prediction)
#     # print(window)

to_predict = df.tail(7)
# to_predict.drop([to_predict.index[-1]],axis=0, inplace=True)
# print(to_predict)


to_predict = np.array(to_predict)
# print(to_predict)


scaled_to_predict = scaler.transform(to_predict)

y_pred = model.predict(np.array([scaled_to_predict]))
print("The predicted numbers in the last lottery game are:", scaler.inverse_transform(y_pred).astype(int)[0])

prediction = df.tail(1)
prediction = np.array(prediction)
print("The actual numbers in the last lottery game were:", prediction[0])
# print('foo')







