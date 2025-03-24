import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def generate_data(NSAMPLES = 10, NHISTO = 100, freq = 10, t_sample = 0.001):
       
    np_times = np.arange( 0, (NSAMPLES + NHISTO) * t_sample, t_sample)   # in seconds
    
    np_sin = np.sin( (2*np.pi * freq) * np_times )
    
    np_sin_windowed = sliding_window_view(np_sin, NHISTO)
    
    return np_sin, np_times, np_sin_windowed
    
    
if __name__ == "__main__":
    np_sin, np_times, np_sin_windowed = generate_data()
    
    import matplotlib.pyplot as plt
    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(np_sin)
    
    for window in np_sin_windowed:
        plt.plot(window)
    
    plt.savefig("plots/" + "sine_data.png")
    plt.show()
    
    
    