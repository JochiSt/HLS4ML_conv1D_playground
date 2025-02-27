"""
    script to convert the network using HLS4ML in HDL code
"""

import os
os.environ["PATH"] = "/tools/Xilinx/Vitis_HLS/2024.2/bin/:" + os.environ["PATH"]
os.environ["PATH"] = "/tools/Xilinx/Vivado/2024.2/bin/:" + os.environ["PATH"]

# Convert the model to FPGA firmware with hls4ml
# Make an hls4ml config & model
import hls4ml
import json
import sys
import numpy as np
import matplotlib.pyplot as plt

from HGQ.layers import HDense, HDenseBatchNorm, HQuantize

# use the already trained model
from tensorflow.keras.models import load_model
from plotting import print_dict


################################################################################
profiling_plots = False

XILINX_PART_NO = "xc7a100t-csg324-1"
################################################################################


################################################################################
from simple_linear_model import create_datasets
x_train, x_test, x_validate, y_train, y_test, y_validate = create_datasets(1000)
x_test = np.concatenate((x_train, x_test, x_validate))
y_test = np.concatenate((y_train, y_test, y_validate))

model = load_model("models/SimpleLinearModel_HGQ_proxy.h5")

################################################################################
# convert model to HLS one
from hls4ml.converters import convert_from_keras_model
hls_model = convert_from_keras_model(model, backend='vitis',output_dir="model_2" ,part=XILINX_PART_NO)

################################################################################
# evaluate the performance of the neural network
hls_model.compile()

################################################################################
y_hls = hls_model.predict(x_test)


################################################################################

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="plots/HLS_HGQ_model.png")

hls_model.build(reset=False, csim=True, synth=True, cosim=False, validation=False, export=True, vsynth=False)

vivado_report = hls4ml.report.parse_vivado_report("model_2")

print_dict(vivado_report)

print(vivado_report)


