import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, CuDNNLSTM, LSTM, Dropout

from dataprocessing import DataProcessor

## DATA

dp = DataProcessor("D:\\leont\\Documents\\Schule\\W-Seminar\\test_v2\\")
SEQUENCE_LENGTH = 50

## MODEL

model = Sequential()

model.add(CuDNNLSTM(88, input_shape=(None, 88), return_sequences=True))
model.add(Dropout(0.2))

model.add(CuDNNLSTM(88))
model.add(Dropout(0.2))

model.add(Dense(88, activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(88, activation="softmax"))

print(model.summary(90))

optimizer = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)

model.compile(loss="categorical_crossentropy",
    optimizer=optimizer,
    metrics=["accuracy"])

model.fit_generator(dp.train_generator_no_padding(SEQUENCE_LENGTH), steps_per_epoch=1000, epochs=30, verbose=1)


############# MODEL HAS TO BE TRANSFORMED AFTER TRAINING,
############# SO IT CAN RUN ON BOTH CPU AND GPU