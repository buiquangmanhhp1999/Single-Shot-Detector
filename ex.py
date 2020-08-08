import numpy as np

array = np.array([[1, 3], [5, 4]])
a = np.argmax(array, axis=1)
print(array[[0, 1], [1, 0]])