"""
    script to convert the network using HLS4ML in HDL code
"""

import os
os.environ["PATH"] = "/tools/Xilinx/Vitis_HLS/2024.2/bin/:" + os.environ["PATH"]
os.environ["PATH"] = "/tools/Xilinx/Vivado/2024.2/bin/:" + os.environ["PATH"]

# Convert the model to FPGA firmware with hls4ml
# Make an hls4ml config & model
import hls4ml
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

config = hls4ml.utils.config_from_keras_model(model, granularity='name')
cfg_q = hls4ml.converters.create_config(backend='Vitis')
cfg_q['IOType'] = 'io_stream'  # Must set this if using CNNs!
cfg_q['KerasModel'] = model
cfg_q['HLSConfig'] = config
cfg_q['OutputDir'] = 'model_2/'
cfg_q['XilinxPart'] = XILINX_PART_NO
cfg_q['Part'] = cfg_q['XilinxPart']

print_dict(cfg_q)

from hls4ml.converters import convert_from_keras_model
hls_model = convert_from_keras_model(
    model, 
    backend='vitis',
    output_dir="model_2",
    part=XILINX_PART_NO,
    hls_config=cfg_q
    )

################################################################################
# evaluate the performance of the neural network
hls_model.compile()

################################################################################
y_hls = hls_model.predict(x_test)
y_keras = model.predict(x_test)

# create plots
fig, ax1 = plt.subplots()
ax1.plot(x_test, y_hls+0.1, "b.", label="HLS + 0.1")
ax1.plot(x_test, y_keras, "g.", label="KERAS HGQ")
ax1.set_ylabel("Prediction")
ax1.set_xlabel("X test")

ax2 = ax1.twinx()
y_hls = y_hls.flatten()
ax2.plot(x_test, y_hls - np.abs(x_test), "r.", label="HLS - X")
ax2.set_ylabel("HLS - X")

fig.tight_layout()
plt.savefig("plots/" + model.name + "_keras_hls_comp.png")
plt.show()

################################################################################
#import sys
#sys.exit(0)

hls4ml.utils.plot_model(hls_model, show_shapes=True, show_precision=True, to_file="plots/HLS_HGQ_model.png")
hls_model.build(reset=False, csim=True, synth=True, cosim=False, validation=False, export=True, vsynth=False)

vivado_report = hls4ml.report.parse_vivado_report("model_2")

print_dict(vivado_report)


