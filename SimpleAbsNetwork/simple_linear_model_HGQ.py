import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import keras
from HGQ.layers import HDense, HDenseBatchNorm, HQuantize, Signature
from HGQ import ResetMinMax, FreeBOPs
from HGQ import get_default_kq_conf, get_default_paq_conf, set_default_kq_conf, set_default_paq_conf
 
def setupModel(fixed_bit_inout = False):
    
    paq_conf = get_default_paq_conf()
    paq_conf['skip_dims'] = 'all'
    paq_conf['rnd_strategy'] = 'fast_uniform_noise_injection'
    paq_conf['init_bw'] = 10
    set_default_paq_conf(paq_conf)

    kq_conf = get_default_kq_conf()
    kq_conf['init_bw'] = 8
    set_default_kq_conf(kq_conf)
    
    inputs = tf.keras.Input(shape=(1,), name="datainput")
    if not fixed_bit_inout:
        # train the input bitwidth
        x = HQuantize(beta=1.e-8)(inputs)
    else:
        # input is fixed to 20 bits (one integer)
        x = Signature( bits=20, int_bits=1, keep_negative=1)(inputs)
    layer_cnt = 0

    x = HDense(16, beta=1.e-8, activation='relu', name="layer_%d" % (layer_cnt))(x)
    layer_cnt += 1

    x = HDense(16, beta=1.e-8, activation='relu', name="layer_%d" % (layer_cnt))(x)
    layer_cnt += 1

    # final layer
    outputs = HDense(1, beta=1.e-8, name="output")(x)
    
    if fixed_bit_inout:
        outputs = Signature( bits=12, int_bits=1, keep_negative=1)(outputs)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="SimpleLinearStream_HGQ")
    model.summary()
    
    keras.utils.plot_model(model, to_file='plots/HGQmodel.png', show_shapes=True)

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


def training(model, SAMPLES=1000000, TRAINING_EPOCHS = 50):
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
    from keras.callbacks import LearningRateScheduler
    from keras.experimental import CosineDecay
        
    _sched = CosineDecay(1e-3, TRAINING_EPOCHS * 1.1)
    sched = LearningRateScheduler(_sched)

    callbacks = [sched, ResetMinMax(), FreeBOPs()]
   
    history = model.fit(
        x_train,                # X training data
        y_train,                # Y training data
        epochs=TRAINING_EPOCHS, # how long do we want to train
        batch_size=500,         # how large is one batch
        shuffle=True,
        validation_data=(x_validate, y_validate),
        callbacks=callbacks     # callbacks (see above)
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
    
def getLayerBitWidth(model):
    """
        get the bit width from each layer
    """
    config = model.get_config()
    for layer in config['layers']:
        print(layer['name'])
        
        act_bw = model.get_layer(layer['name']).act_bw.numpy()
        print(act_bw)

if __name__ == "__main__":
    model = setupModel(fixed_bit_inout=True)
    training(model, TRAINING_EPOCHS=50)
    #getLayerBitWidth(model)
