import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM, Dropout

from dataprocessing import DataProcessor

## DATA

dp = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")

## MODEL

model = Sequential()

model.add(CuDNNLSTM(88, batch_input_shape=(1, None, 88), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(88))
model.add(Dropout(0.2))

model.add(Dense(88, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(88, activation="sigmoid"))

print(model.summary(90))

optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

model.fit_generator(dp.train_generator(), steps_per_epoch=2, epochs=10, verbose=1)


############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU