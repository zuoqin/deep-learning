import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
from keras.layers import LSTM
def adj_r2_score(r2, n, k):
    return 1-((1-r2)*((n-1)/(n-k-1)))



exchange_data = pd.read_csv('XAU_USD.csv', sep=';')
exchange_data["Date"] = pd.to_datetime(exchange_data["Date"])
ind_exchange_data = exchange_data.set_index(["Date"], drop=True)


data_frame = ind_exchange_data.sort_index(axis=1 ,ascending=True)
data_frame = data_frame.iloc[::-1]
df = data_frame[["Price"]]
split_date = pd.Timestamp('01-01-2016')

train = df.loc[:split_date]
test = df.loc[split_date:]

sc = MinMaxScaler()
train_sc = sc.fit_transform(train)
test_sc = sc.transform(test)

X_train = train_sc[:-1]
y_train = train_sc[1:]

X_test = test_sc[:-1]
y_test = test_sc[1:]


X_tr_t = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_tst_t = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])




#
# K.clear_session()
# model_lstm = Sequential()
# model_lstm.add(LSTM(7, input_shape=(1, X_train.shape[1]), activation='relu', kernel_initializer='lecun_uniform', return_sequences=False))
# model_lstm.add(Dense(1))
# model_lstm.compile(loss='mean_squared_error', optimizer='adam')
# early_stop = EarlyStopping(monitor='loss', patience=5, verbose=1)
# history_model_lstm = model_lstm.fit(X_tr_t, y_train, epochs=200, batch_size=1, verbose=1, shuffle=False, callbacks=[early_stop])
#
# y_pred_test_lstm = model_lstm.predict(X_tst_t)
# y_train_pred_lstm = model_lstm.predict(X_tr_t)
# print("The R2 score on the Train set is:\t{:0.3f}".format(r2_score(y_train, y_train_pred_lstm)))
# r2_train = r2_score(y_train, y_train_pred_lstm)
# print("The Adjusted R2 score on the Train set is:\t{:0.3f}\n".format(adj_r2_score(r2_train, X_train.shape[0], X_train.shape[1])))
# print("The R2 score on the Test set is:\t{:0.3f}".format(r2_score(y_test, y_pred_test_lstm)))
# r2_test = r2_score(y_test, y_pred_test_lstm)
# print("The Adjusted R2 score on the Test set is:\t{:0.3f}".format(adj_r2_score(r2_test, X_test.shape[0], X_test.shape[1])))
#
# model_lstm.save('LSTM_NonShift.h5')

model_lstm = load_model('LSTM_NonShift.h5')

score_lstm= model_lstm.evaluate(X_tst_t, y_test, batch_size=1)

print('LSTM: %f'%score_lstm)

y_pred_test_LSTM = model_lstm.predict(X_tst_t)

col1 = pd.DataFrame(y_test, columns=['True'])
col3 = pd.DataFrame(y_pred_test_LSTM, columns=['LSTM_prediction'])
#col5 = pd.DataFrame(history_model_lstm.history['loss'], columns=['Loss_LSTM'])
results = pd.concat([col1, col3], axis=1)
results.to_excel('PredictionResults_LSTM_NonShift.xlsx')

plt.plot(y_test, label='True')
plt.plot(y_pred_test_LSTM, label='LSTM')
plt.title("LSTM's_Prediction")
plt.xlabel('Observation')
plt.ylabel('INR_Scaled')
plt.legend()
plt.show()

