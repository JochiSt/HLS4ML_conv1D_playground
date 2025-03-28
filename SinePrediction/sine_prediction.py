import tensorflow as tf
from keras.optimizers import Adam

import numpy as np

import matplotlib.pyplot as plt

from sine_data import create_datasets, generate_data


def setupModel():
    inputs = tf.keras.Input(shape=(100,1,), name="datainput")
    layer_cnt = 0

    x = tf.keras.layers.Dense(100, name="dense_%d" % (layer_cnt))(inputs)
    layer_cnt += 1
    
    x = tf.keras.layers.Conv1D(32,
                              3,
                              use_bias = True,
                              strides=1,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = tf.keras.layers.Conv1D(32,
                              3,
                              strides=1,
                              use_bias = False,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    layer_cnt += 1

    x = tf.keras.layers.Conv1D(16,
                              3,
                              use_bias = True,
                              strides=1,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = tf.keras.layers.Conv1D(16,
                              3,
                              strides=1,
                              use_bias = False,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    layer_cnt += 1
    
    x = tf.keras.layers.Conv1D(8,
                              3,
                              use_bias = True,
                              strides=1,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)    
    layer_cnt += 1
    
    x = tf.keras.layers.Conv1D(8,
                              3,
                              strides=1,
                              use_bias = False,
                              padding = "same",
                              kernel_initializer = "glorot_uniform",
                              name="conv1d_%d"%(layer_cnt)
                              )(x)
    layer_cnt += 1    
    
    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(32, name="dense_%d" % (layer_cnt))(x)
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)
    layer_cnt += 1

    x = tf.keras.layers.Dense(16, name="dense_%d" % (layer_cnt))(x)
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)
    layer_cnt += 1

    # final layer
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SinePrediction")
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file="plots/"+model.name+".png", show_shapes=True)

    # Compile the model using the standard 'adam' optimizer and
    # the mean squared error or 'mse' loss function for regression.
    # the mean absolute error or 'mae' is also used as a metric
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", metrics=["mae", "mse"])

    return model

def training(model, SAMPLES=100000, epochs=50):
    print("Generating data ...")
    y_train, y_test, y_validate, x_train, x_test, x_validate = create_datasets(SAMPLES)

    # Double check that our splits add up correctly
    #assert (len(x_train) + len(x_validate) + len(x_test)) == SAMPLES

    print("Points using for:")
    print(" - Training:  ", len(x_train))
    print(" - Validation:", len(x_validate))
    print(" - Testing:   ", len(x_test))
    print("in total:     ", SAMPLES)

    x_train     = x_train.reshape(      (np.shape(x_train)[0]    , np.shape(x_train)[1]   ,1 ) )
    x_test      = x_test.reshape(       (np.shape(x_test)[0]     , np.shape(x_test)[1]    ,1 ) )
    x_validate  = x_validate.reshape(   (np.shape(x_validate)[0] , np.shape(x_validate)[1],1 ) )
    
    print("X Train", np.shape(x_train))
    print("Y Train", np.shape(y_train))
    
    ###########################################################################
    # Train the network
    # fully train the network
    from keras.callbacks import LearningRateScheduler
    from keras.experimental import CosineDecay    
    _sched = CosineDecay(1e-2, epochs * 1.02)
    sched = LearningRateScheduler(_sched)

    callbacks = [sched]    
    
    history = model.fit(
        x_train,
        y_train,
        callbacks=callbacks,
        epochs=epochs,   # how long do we want to train
        batch_size=500,  # how large is one batch
        shuffle=True,
        validation_data=(x_validate, y_validate),
    )

    ###########################################################################
    # create typical deep learning performance plots
    # Plot the training history
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    # the MAE mean absolute error is a good quantity, which gives the
    # 'accuracy' of our model.
    mae = history.history["mae"]
    val_mae = history.history["val_mae"]

    # create x-axis
    epochs = range(1, len(loss) + 1)

    # create plots
    fig, ax1 = plt.subplots()
    ax1.plot(epochs, loss, "b", label="Training loss")
    ax1.plot(epochs, val_loss, "b--", label="Validation loss")

    ax1.set_yscale("log")

    ax2 = ax1.twinx()
    ax2.plot(epochs, mae, "r", label="Training MAE")
    ax2.plot(epochs, val_mae, "r--", label="Validation MAE")

    ax2.set_yscale("log")

    ################################################################################
    # tune the look-a-like of the plot

    ax1.set_xlabel("epochs")
    ax1.set_ylabel("loss")
    ax2.set_ylabel("MAE")
    plt.title("Training and validation Performance")

    for label in ax1.get_yticklabels():
        label.set_color("b")
    ax1.yaxis.label.set_color("b")

    for label in ax2.get_yticklabels():
        label.set_color("r")
    ax2.yaxis.label.set_color("r")

    fig.legend(framealpha=1)
    fig.tight_layout()
    plt.savefig("plots/" + model.name + "_train_perf.png")
    plt.show()

    # save network
    model.save("models/"+model.name+".keras")
    model.save("models/"+model.name+".h5")

def plot_data(x, y, y_pred=None, model_name=""):
    plt.clf()
    plt.title("Comparison of predictions to actual values")
    
    for i, x_dat in enumerate(x):
        plt.plot(np.arange(len(x_dat)), x_dat, "b--", label="input")
        plt.plot(len(x_dat)+1, y[i], "go", label="expectation")
    
        if y_pred is not None:
            plt.plot(len(x_dat) + 1, y_pred[i], "r.", label="Prediction")
            
    #plt.legend(framealpha=1)
    plt.savefig("plots/"+model_name+"_evaluation.png")
    plt.show()
    
if __name__ == "__main__":
    train = False
    
    train = True
    
    if train:
        ########################################################################
        # create model
        model = setupModel()
        
        
        ########################################################################
        # train model
        training(model, SAMPLES=10000, epochs=50)
        
    else:
        model = tf.keras.models.load_model('models/SimpleLinearModel.keras')

    ############################################################################
    # evaluate model and create some nice plots
    
    np_sin_windowed, np_times_windowed, prediction, (f,p,a) = generate_data(NSAMPLES = 10000, NHISTO = 100, freq = [2,20], phase=[0, 2* np.pi], t_sample = 0.001)
    
    np_sin_windowed /= 2
    np_sin_windowed += 0.5
    
    prediction /= 2
    prediction += 0.5

    np_sin_windowed = np_sin_windowed.reshape(   (np.shape(np_sin_windowed)[0]  ,  np.shape(np_sin_windowed)[1] ,1 ))
    
    predictions = model.predict(np_sin_windowed)
    predictions = predictions.reshape( (np.size(predictions)))
        
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
    