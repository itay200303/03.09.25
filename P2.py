import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

X = np.array([1,2,3,4,5,6,7,8,9,10], dtype=float)
y = np.array([0,0,0,0,1,1,1,1,1,1], dtype=float)

model = Sequential()
model.add(Dense(units=1, input_shape=[1], activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.fit(X, y, epochs=500, verbose=0)

probability = model.predict([6.5])[0][0]
print(f"chance to get disscont: {probability:.2f}")
print("yes" if probability >= 0.5 else "no")