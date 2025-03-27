
import numpy as np

np_sin = np.array( [ [ 1,2,3], [4,5,6] ])

print(np_sin)

print(np.size(np_sin))

np_noise = np.random.normal(0, 0.05, np.size(np_sin))
np_noise = np_noise.reshape( np.shape(np_sin) )
print(np_noise)

np_sin = np_sin + np_noise
    
print(np_sin)