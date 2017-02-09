# This file uses a FLIC trained model and applies it to a video sequence from Poses in the Wild
#
# Download the model:
#    wget http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel -P ../../models/heatmap-flic-fusion/


import sys
sys.path.append("../")

from collections import namedtuple
import numpy

import applyNet

# Options
opt = namedtuple('Options', 'visualise, useGPU, dims, numJoints, layerName, modelDefFile, modelFile, inputDir, numFiles')
opt.visualise = True # Visualise predictions
opt.useGPU = True # Run on GPU
opt.dims = [256, 256] # Input dimensions
opt.numJoints = 7 # Number of joints
opt.layerName = 'conv5_fusion' # Output layer name
opt.modelDefFile = '../../models/heatmap-flic-fusion/matlab.prototxt' # Model definition
opt.modelFile = '../../models/heatmap-flic-fusion/caffe-heatmap-flic.caffemodel' # Model weights    
opt.inputDir = 'my_sample/' # Image input directory
opt.numFiles = 29 # Number of input images

# Create image file list
imInds = numpy.arange(1, opt.numFiles+1)
files = []
for ind in imInds:
    files.append(str(ind)+".png")

# Apply network
joints = applyNet.applyNet(files, opt)
 
