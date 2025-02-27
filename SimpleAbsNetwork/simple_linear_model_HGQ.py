import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import keras
from HGQ.layers import HDense, HDenseBatchNorm, HQuantize
from HGQ import ResetMinMax, FreeBOPs
 
def setupModel():
    inputs = tf.keras.Input(shape=(1,), name="datainput")
    x = HQuantize(beta=1.e-7)(inputs)
    layer_cnt = 0

    x = HDense(16, beta=1.e-7, activation='relu', name="layer_%d" % (layer_cnt))(x)
    layer_cnt += 1

    x = HDense(16, beta=1.e-7, activation='relu', name="layer_%d" % (layer_cnt))(x)
    layer_cnt += 1

    # final layer
    outputs = HDense(1, beta=1.e-7, name="output")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SimpleLinearStream_HGQ")
    model.summary()

    # Compile the model using the standard 'adam' optimizer and
    # the mean squared error or 'mse' loss function for regression.
    # the mean absolute error or 'mae' is also used as a metric
    model.compile(optimizer="adam", loss="mse", metrics=["mae", "mse"])

    return model


def generate_data(NSAMPLES=10000):
    """
    generate training / validation data for the network below
    """
    # Generate some random samples
    x_values = np.random.uniform(low=-1, high=1, size=NSAMPLES)

    y_values = np.abs(x_values) + np.random.normal(loc = 0, scale=0.05, size=len(x_values))
    return x_values, y_values


def create_datasets(SAMPLES=10000):
    # We'll use 60% of our data for training and 20% for testing.
    # The remaining 20% will be used for validation. Calculate the indices of
    # each section.
    TRAIN_SPLIT = int(0.6 * SAMPLES)
    TEST_SPLIT = int(0.2 * SAMPLES + TRAIN_SPLIT)

    # generate the training data
    x_values, y_values = generate_data(SAMPLES)

    # convert data into numpy arrays
    x_values = np.array(x_values)  # waveforms
    y_values = np.array(y_values)  # parameters

    # Use np.split to chop our data into three parts.
    # The second argument to np.split is an array of indices where the data
    # will be split. We provide two indices, so the data will be divided into
    # three chunks.
    x_train, x_test, x_validate = np.split(x_values, [TRAIN_SPLIT, TEST_SPLIT])
    y_train, y_test, y_validate = np.split(y_values, [TRAIN_SPLIT, TEST_SPLIT])

    return x_train, x_test, x_validate, y_train, y_test, y_validate


def training(model, SAMPLES=1000000):
    x_train, x_test, x_validate, y_train, y_test, y_validate = create_datasets(SAMPLES)

    # Double check that our splits add up correctly
    assert (len(x_train) + len(x_validate) + len(x_test)) == SAMPLES

    print("Points using for:")
    print(" - Training:  ", len(x_train))
    print(" - Validation:", len(x_validate))
    print(" - Testing:   ", len(x_test))
    print("in total:     ", SAMPLES)

    ###########################################################################
    # Train the network
    # fully train the network
    
    callbacks = [ResetMinMax(), FreeBOPs()]
   
    history = model.fit(
        x_train,
        y_train,
        epochs=50,  # how long do we want to train
        batch_size=500,  # how large is one batch
        shuffle=True,
        validation_data=(x_validate, y_validate),
        callbacks=callbacks
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

    from HGQ import trace_minmax, to_proxy_model

    trace_minmax(model, x_train, verbose=True, cover_factor=1.0)
    proxy = to_proxy_model(model, aggressive=True)

    # save network
    model.save("models/SimpleLinearModel_HGQ.keras")
    model.save("models/SimpleLinearModel_HGQ.h5")
    
    proxy.save("models/SimpleLinearModel_HGQ_proxy.keras")
    proxy.save("models/SimpleLinearModel_HGQ_proxy.h5")

    ###########################################################################
    # Test the network
    # Plot predictions against actual values
    predictions = proxy.predict(x_test)

    plt.clf()
    plt.title("Comparison of predictions to actual values")
    plt.plot(x_test, y_test, "b.", label="Actual")
    plt.plot(x_test, predictions, "r.", label="Prediction")
    plt.legend(framealpha=1)
    plt.savefig("plots/" + proxy.name + "HGQ_proxy_prediction.png")
    plt.show()


if __name__ == "__main__":
    model = setupModel()
    training(model)
