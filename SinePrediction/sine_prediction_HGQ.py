"""
"""

import tensorflow as tf
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

import keras
from HGQ.layers import HDense, HConv1D, HQuantize, Signature, HActivation, PFlatten
from HGQ import ResetMinMax, FreeBOPs
from HGQ import get_default_kq_conf, get_default_paq_conf, set_default_kq_conf, set_default_paq_conf
 
def setupModel():
    inputs = tf.keras.Input(shape=(100,1,), name="datainput")
    layer_cnt = 0

    x = Signature( bits=20, int_bits=1, keep_negative=1)(inputs)

    #x = HDense(100, beta=1e-8, name="dense_%d" % (layer_cnt))(x)
    #layer_cnt += 1
    
    x = HConv1D(8,
                3,
                beta=1e-8,
                use_bias = True,
                strides=1,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    x = HActivation("relu", beta=1e-8, name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = HConv1D(8,
                3,
                beta=1e-8,                
                strides=1,
                use_bias = False,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    layer_cnt += 1

    x = HConv1D(8,
                3,
                beta=1e-8,                
                use_bias = True,
                strides=1,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    x = HActivation("relu", beta=1e-8, name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = HConv1D(8,
                3,
                beta=1e-8,                
                strides=1,
                use_bias = False,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    layer_cnt += 1
    
    x = HConv1D(8,
                3,
                beta=1e-8,                
                use_bias = True,
                strides=1,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    x = HActivation("relu", beta=1e-8, name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = HConv1D(8,
                3,
                beta=1e-8,                
                strides=1,
                use_bias = False,
                padding = "same",
                kernel_initializer = "glorot_uniform",
                name="conv1d_%d"%(layer_cnt)
                )(x)
    layer_cnt += 1    
    
    x = PFlatten()(x)

    x = HDense(32, beta=1e-8, name="dense_%d" % (layer_cnt))(x)
    x = HActivation("relu", beta=1e-8, name="activation_%d"%(layer_cnt))(x)
    layer_cnt += 1

    x = HDense(16, beta=1e-8, name="dense_%d" % (layer_cnt))(x)
    x = HActivation("relu", beta=1e-8, name="activation_%d"%(layer_cnt))(x)
    layer_cnt += 1

    # final layer
    outputs = HDense(1, beta=1e-8, name="output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SinePredictionHGQ")
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file="plots/"+model.name+".png", show_shapes=True)

    # Compile the model using the standard 'adam' optimizer and
    # the mean squared error or 'mse' loss function for regression.
    # the mean absolute error or 'mae' is also used as a metric
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae", "mse"])

    return model

if __name__ == "__main__":
    train = False
    
    #train = True
    
    if train:
        ########################################################################
        # create model
        model = setupModel()
        
        
        ########################################################################
        # train model
        from sine_prediction import training
        model, (x_train, y_train) = training(model, SAMPLES=10000, epochs=50, addcallbacks=[ResetMinMax(), FreeBOPs()])
        
        from HGQ import trace_minmax, to_proxy_model
        trace_minmax(model, x_train, verbose=True, cover_factor=1.0)
        proxy = to_proxy_model(model, aggressive=True)
                
        proxy._name = model.name
                
        proxy.save("models/"+model.name+"_proxy.h5")
        proxy.save("models/"+model.name+"_proxy.keras")
        
    else:
        model = tf.keras.models.load_model('models/SinePredictionHGQ_proxy.keras')

    ############################################################################
    # evaluate model and create some nice plots
    from sine_data import generate_data

    np_sin_windowed, np_times_windowed, prediction, (f,p,a) = generate_data(NSAMPLES = 10000, NHISTO = 100, freq = [2,20], phase=[0, 2* np.pi], t_sample = 0.001)
    
    np_sin_windowed /= 2
    np_sin_windowed += 0.5
    
    prediction /= 2
    prediction += 0.5

    np_sin_windowed = np_sin_windowed.reshape(   (np.shape(np_sin_windowed)[0]  ,  np.shape(np_sin_windowed)[1] ,1 ))
    
    predictions = model.predict(np_sin_windowed)
    predictions = predictions.reshape( (np.size(predictions)))
    
    from sine_prediction import plot_data
    plot_data( np_sin_windowed[:10], prediction[:10], predictions[:10], model_name = model.name)

    difference = prediction - predictions
    
    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(f, difference, ".", label="difference")            
    plt.savefig("plots/"+model.name+"_difference_vs_f.png")
    plt.show()
    
    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(p, difference, ".", label="difference")            
    plt.savefig("plots/"+model.name+"_difference_vs_p.png")
    plt.show()    

    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(a, difference, ".", label="difference")            
    plt.savefig("plots/"+model.name+"_difference_vs_a.png")
    plt.show()
    