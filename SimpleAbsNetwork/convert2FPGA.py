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

################################################################################
profiling_plots = False
################################################################################

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
    
config['Model']['TraceOutput'] = True
    
################################################################################
# change the configuration and adapt bitwidth
config['LayerName']['layer_0']['Precision']['weight'] = 'ap_fixed<8,2>'
config['LayerName']['layer_0']['Precision']['bias'] = 'ap_fixed<8,2>'

config['LayerName']['activation']['Precision']['result'] = 'ap_fixed<8,2>'
config['LayerName']['activation']['TableSize'] = 0

config['LayerName']['layer_1']['Precision']['weight'] = 'ap_fixed<8,2>'
config['LayerName']['layer_1']['Precision']['bias'] = 'ap_fixed<8,2>'

config['LayerName']['activation_1']['Precision']['result'] = 'ap_fixed<8,2>'
config['LayerName']['activation_1']['TableSize'] = 0

config['LayerName']['output']['Precision']['weight'] = 'ap_fixed<8,2>'
config['LayerName']['output']['Precision']['bias'] = 'ap_fixed<8,2>'

print("-----------------------------------")
print_dict(config)
print("-----------------------------------")

cfg_q = hls4ml.converters.create_config(backend='Vitis')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['HLSConfig'] = config
cfg_q['KerasModel'] = model
cfg_q['OutputDir'] = 'model_1/'
cfg_q['XilinxPart'] = 'xcu250-figd2104-2L-e'

################################################################################
# convert model to HLS one
hls_model = hls4ml.converters.keras_to_hls(cfg_q)

################################################################################
# evaluate the performance of the neural network
hls_model.compile()
hls4ml_pred, hls4ml_trace = hls_model.trace(x_test[:1000])
keras_trace = get_ymodel_keras(model, x_test[:1000])
y_hls = hls_model.predict(x_test)

# have a look at the differences between the layers (see, where we have the most
# impact of the quantisation)
differences = {}
max_params = 0
for layer in hls4ml_trace.keys():
    print(layer, len(hls4ml_trace[layer]))
    difference = hls4ml_trace[layer] - keras_trace[layer]
    print( np.mean( difference, axis=0) )
    differences[layer] = np.mean(difference, axis=0)
        
    max_params = max(max_params, differences[layer].size )


difference_array = np.full(( len(differences.keys()), max_params ), np.nan)
for i, layer in enumerate(differences.keys()):
    for j,value in enumerate(differences[layer]):
        difference_array[i][j] = value
    
fig = plt.figure()

ax = fig.add_subplot(111)
ax.set_title('HLS - Keras (%s)'%(model.name))

from matplotlib import ticker
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# set proper Y labels
ax.set_yticks(np.arange(len(hls4ml_trace.keys())))
ax.set_yticklabels( list(hls4ml_trace.keys()) )

absolute_maximum = max( abs(np.min(difference_array)), np.max(difference_array) )

from matplotlib.colors import Normalize
from matplotlib import cm
norm = Normalize(vmin=-1*absolute_maximum, vmax=absolute_maximum)
plt.imshow(difference_array, interpolation = 'none', cmap="seismic", norm=norm)
plt.colorbar(cm.ScalarMappable(norm=norm, cmap="seismic"), ax=ax)
plt.show()

if profiling_plots:
    plots = numerical(
        model=model,            # Keras model
        hls_model=hls_model,    # HLS model
        X=x_test                # test data
        )
    plt.show()

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="plots/HLSmodel.png")


