"""
    script to convert the network using HLS4ML in HDL code
"""

import os
os.environ["PATH"] = "/opt/Xilinx/Vitis_HLS/2023.2/bin/:" + os.environ["PATH"]

# Convert the model to FPGA firmware with hls4ml
# Make an hls4ml config & model
import hls4ml
import json
import sys
import numpy as np
import matplotlib.pyplot as plt


# use the already trained model
from tensorflow.keras.models import load_model

model = load_model("models/SimpleLinearModel.h5")

# create configuration for each layer
config = hls4ml.utils.config_from_keras_model(model, granularity='name', backend='Vitis')  
print("-----------------------------------")
from plotting import print_dict
print_dict(config)
print("-----------------------------------")

################################################################################
from simple_linear_model import create_datasets
x_train, x_test, x_validate, y_train, y_test, y_validate = create_datasets(1000)
x_test = np.concatenate((x_train, x_test, x_validate))
y_test = np.concatenate((y_train, y_test, y_validate))

from hls4ml.model.profiling import numerical, get_ymodel_keras

for layer in config['LayerName'].keys():
    config['LayerName'][layer]['Trace'] = True
    
################################################################################
# change the configuration and adapt bitwidth
config['LayerName']['layer_0']['Precision']['weight'] = 'ap_fixed<8,4>'
    
################################################################################
# convert model to HLS one
hls_model = hls4ml.converters.convert_from_keras_model(
    model, hls_config=config, output_dir='model_1/hls4ml_prj_2', part='xcu250-figd2104-2L-e'
)

################################################################################
# evaluate the performance of the neural network
hls_model.compile()
hls4ml_pred, hls4ml_trace = hls_model.trace(x_test[:1000])
keras_trace = get_ymodel_keras(model, x_test[:1000])
y_hls = hls_model.predict(x_test)

# have a look at the differences between the layers (see, where we have the most
# impact of the quantisation)
differences = {}
for layer in hls4ml_trace.keys():
    print(layer, len(hls4ml_trace[layer]))
    difference = hls4ml_trace[layer] - keras_trace[layer]
    print( np.mean( difference, axis=0) )
    differences[layer] = np.mean(difference, axis=0)

if profiling_plots:
    plots = numerical(
        model=model,            # Keras model
        hls_model=hls_model,    # HLS model
        X=x_test                # test data
        )
    plt.show()

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="plots/HLSmodel.png")


