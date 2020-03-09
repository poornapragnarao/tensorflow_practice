import numpy as np
import tensorflow.keras as keras

training_data = np.array([[0,0],[0,1],[1,0],[1,1]], "float32")

target_data = np.array([[0],[1],[1],[0]], "float32")

model = keras.models.Sequential()
model.add(keras.layers.Dense(2, input_dim=2, activation='sigmoid'))
model.add(keras.layers.Dense(1, activation='sigmoid'))

model.compile(loss='mean_squared_error',
              optimizer=keras.optimizers.SGD(lr=0.5),
              metrics=['accuracy'])

model.fit(training_data, target_data, nb_epoch=500, verbose=2, batch_size=20)

print(model.predict(training_data).round())
