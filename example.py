import numpy as np

x = np.linspace(0, 4, 3)
y = np.linspace(0, 5, 8)

xv, yv = np.meshgrid(x, y)
xv = np.expand_dims(xv, axis=-1)
yv = np.expand_dims(yv, axis=-1)

xv = np.tile(xv, (1, 1, 4))
print('x: \n', x)
print('y: \n', y)
print('xv: \n', xv)
print('yv: \n', yv)
print('result: \n', np.tile(xv, (1, 1, 6)))