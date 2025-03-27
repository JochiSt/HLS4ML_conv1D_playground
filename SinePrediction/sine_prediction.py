import tensorflow as tf
from keras.optimizers import Adam

import numpy as np

import matplotlib.pyplot as plt

from sine_data import create_datasets

def ResNetBlockConv1D(x,
                    n_filter = 16,
                    kernel_size = 3,
                    padding = "same",
                    conv_activation = "relu",
                    kernel_initializer = "glorot_uniform",
                    ):
    r"""
    Basic ResNetBlock implementation without Batchnormalization.

    Args:
      x: A keras layers object. e.g. output of tf.keras.layers.Conv1D.
      n_filter: number of convolution filters
      kernel_size: An integer or tuple/list of a single integer, specifying the length of the 1D convolution window.
      padding: One of "valid", "same" or "causal" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input such that output has the same height/width dimension as the input. "causal" results in causal (dilated) convolutions, e.g. output[t] does not depend on input[t+1:].
      conv_activation: activation function of the first convolution layer in the block
      kernel_initializer: initializer for the weights
      activation: activation function which is applied after adding identity to conv block
    Returns:
      A keras layer.
    """

    #two convolution layers in ResNetBlock
    fx = tf.keras.layers.Conv1D(n_filter,
                              kernel_size,
                              activation = conv_activation,
                              use_bias = True,
                              strides=1,
                              padding = padding,
                              kernel_initializer = kernel_initializer)(x)
    fx = tf.keras.layers.Conv1D(n_filter,
                              kernel_size,
                              strides=1,
                              use_bias = False,
                              padding = padding,
                              kernel_initializer = kernel_initializer)(fx)

    x = fx
    return x


def setupModel():
    inputs = tf.keras.Input(shape=(100,1,), name="datainput")
    layer_cnt = 0

    x = ResNetBlockConv1D(inputs)
    x = ResNetBlockConv1D(x)
    x = ResNetBlockConv1D(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(16, name="dense_%d" % (layer_cnt))(x)
    layer_cnt += 1
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)

    x = tf.keras.layers.Dense(16, name="dense_%d" % (layer_cnt))(x)
    layer_cnt += 1
    x = tf.keras.layers.Activation("relu", name="activation_%d"%(layer_cnt))(x)

    # final layer
    outputs = tf.keras.layers.Dense(1, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SinePrediction")
    model.summary()
    
    tf.keras.utils.plot_model(model, to_file="plots/sine_prediction.png", show_shapes=True)

    # Compile the model using the standard 'adam' optimizer and
    # the mean squared error or 'mse' loss function for regression.
    # the mean absolute error or 'mae' is also used as a metric
    model.compile(optimizer=Adam(learning_rate=5e-3), loss="mse", metrics=["mae", "mse"])

    return model

def training(model, SAMPLES=100000, epochs=50):
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
    history = model.fit(
        x_train,
        y_train,
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
    model.save("models/SimpleLinearModel.keras")
    model.save("models/SimpleLinearModel.h5")

def plot_data(x, y, y_pred=None):
    plt.clf()
    plt.title("Comparison of predictions to actual values")
    
    for i, x_dat in enumerate(x):
        plt.plot(np.arange(len(x_dat)), x_dat, "b--", label="input")
        plt.plot(len(x_dat)+1, y[i], "go", label="expectation")
    
        if y_pred is not None:
            plt.plot(len(x_dat) + 1, y_pred[i], "r.", label="Prediction")
            
    #plt.legend(framealpha=1)
    plt.savefig("plots/evaluation_data.png")
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
        training(model, SAMPLES=10000, epochs=2)
        
    else:
        model = tf.keras.models.load_model('models/SimpleLinearModel.keras')

    ############################################################################
    # evaluate model and create some nice plots
    _, _, y_validate, _, _, x_validate = create_datasets(5, split=False)

    x_validate = x_validate.reshape(   (np.shape(x_validate)[0]  ,  np.shape(x_validate)[1] ,1 ))
    
    predictions = model.predict(x_validate)
        
    plot_data( x_validate, y_validate, predictions)