import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

def generate_data(NSAMPLES = 10, NHISTO = 100, freq = 10, t_sample = 0.001):
    """
        NSAMPLES    = number of waveforms, which should be generated
        NHISTO      = number of points per waveform
        freq        = frequency of the sine in Hz
        t_sample    = sampling time = 1 / sampling frequency
    """
       
    np_times = np.arange( 0, (NSAMPLES + NHISTO + 1) * t_sample, t_sample)   # in seconds
    
    np_sin = np.sin( (2*np.pi * freq) * np_times )
    
    indexes = range(len(np_sin)-1)
    indexes_windowed = sliding_window_view(indexes, NHISTO)
    
    np_sin_windowed = np_sin[indexes_windowed]
    
    np_noise = np.random.normal(0, 0.05, np.size(np_sin_windowed))
    np_noise = np_noise.reshape( np.shape(np_sin_windowed) )
    np_sin_windowed = np_sin_windowed + np_noise
    
    np_times_windowed = np_times[indexes_windowed]
    prediction = np_sin[ np.max(indexes_windowed, axis=1) +1 ]
    
    return np_sin_windowed, np_times_windowed, prediction
    
    
def create_datasets(SAMPLES=10000, split=True):
    # We'll use 60% of our data for training and 20% for testing.
    # The remaining 20% will be used for validation. Calculate the indices of
    # each section.
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

    # generate the training data
    y_values, _, x_values = generate_data(NSAMPLES=SAMPLES)

    # convert data into numpy arrays
    x_values = np.array(x_values)  # waveforms
    y_values = np.array(y_values)  # parameters

    # Use np.split to chop our data into three parts.
    # The second argument to np.split is an array of indices where the data
    # will be split. We provide two indices, so the data will be divided into
    # three chunks.
    if split:
        x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
        y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])
    else:
        x_train = x_values
        x_test = x_values
        x_validate = x_values
        
        y_train = y_values
        y_test = y_values
        y_validate = y_values

    return x_train, x_test, x_validate, y_train, y_test, y_validate
    
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
    
    