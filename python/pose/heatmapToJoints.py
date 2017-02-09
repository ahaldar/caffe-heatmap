# Find joints in heatmap (== max locations in heatmap)

import numpy

def heatmapToJoints(heatmapResized, numJoints, useMin = None):
    joints = numpy.zeros((2, numJoints))
    for i in numpy.arange(0, numJoints):
        sub_img = heatmapResized[i, :, :]
        vec = sub_img.flatten()
        val, idx = numpy.amax(vec), numpy.argmax(vec)
        #print (val, idx)
        (y, x) = numpy.unravel_index(idx, sub_img.shape)
        #print (y,x)
        joints[:, i] = (x, y)
    return joints
