################################################################
# Startup/header code for using TensorFlow v1 for examples in
# Charniak's Introduction to Deep Learning book
#
# Author: R. Zanibbi
# Author: E. Lima
################################################################

import tensorflow
import os

# Disable eager execution to permit "Session" objects to run
tensorflow.compat.v1.disable_eager_execution()

# Redefine 'tf' to v1 of Tensorflow to match examples
tf = tensorflow.compat.v1

# Remove startup message - WARNING -- not sure if this is the best log level
# setting.
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
