import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def generate_data(NSAMPLES = 10, NHISTO = 100, freq = 10, t_sample = 0.001):
       
    np_times = np.arange( 0, (NSAMPLES + NHISTO + 1) * t_sample, t_sample)   # in seconds
    
    np_sin = np.sin( (2*np.pi * freq) * np_times )
    
    indexes = range(len(np_sin)-1)
    indexes_windowed = sliding_window_view(indexes, NHISTO)
    
    np_sin_windowed = np_sin[indexes_windowed]
    np_times_windowed = np_times[indexes_windowed]
    prediction = np_sin[ np.max(indexes_windowed, axis=1) +1 ]
    
    return np_sin_windowed, np_times_windowed, prediction
    
    
if __name__ == "__main__":
    
    np_sin, np_times, prediction = generate_data()
        
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title("Comparison of predictions to actual values")

    for i, sine in enumerate(np_sin):
        plt.plot(np_times[i], sine + i/10)
        plt.plot( np.max( np_times[i]) + 0.001, prediction[i] + i/10, "*")
    
    plt.savefig("plots/" + "sine_data.png")
    plt.show()
    
    