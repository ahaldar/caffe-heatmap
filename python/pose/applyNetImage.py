# Apply network to a single image

import time
import scipy.misc
import numpy
import matplotlib.pyplot
import matplotlib.image

import prepareImagePose
import processHeatmap
import getConfidenceImage
import plotSkeleton

def applyNetImage(imgFile, net, opt):
    # Read & reformat input image
    img=scipy.misc.imread(opt.inputDir+imgFile)
    input_data = prepareImagePose.prepareImagePose(img, opt)

    # Forward pass
    tic = time.time()
    net.blobs['data'].data[...] = input_data
    net.forward()
    features = net.blobs[opt.layerName].data
    joints, heatmaps = processHeatmap.processHeatmap(features, opt)
    toc = time.time()
    print(toc-tic)

    # Visualisation
    if opt.visualise:
        # Heatmap
        heatmapVis = getConfidenceImage.getConfidenceImage(heatmaps, img)
        #figure(2)
        matplotlib.pyplot.imshow(heatmapVis)
        matplotlib.pyplot.show()
        
        # Original image overlaid with joints
        #figure(1)
        matplotlib.pyplot.imshow(img)
        plotSkeleton.plotSkeleton(joints, [], [])
        matplotlib.pyplot.show()
