import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([900, 1200, 1500, 1800, 2100, 2400, 2700], dtype=float)

model = Sequential()
model.add(Dense(units=1, input_shape=[1], activation='linear'))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X, y, epochs=500, verbose=0)

prediction = model.predict([5.5])
print(f"prediction{prediction[0][0]:.2f}")