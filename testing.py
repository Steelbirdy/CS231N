import numpy as np
import timer

a = np.arange(1, 10).reshape((3, 3))
print(np.max(a, axis=1, keepdims=True))