
# Reformat output heatmap: rotate & permute color channels

import numpy
#from skimage.transform import resize
import scipy.misc

import heatmapToJoints

def processHeatmap(heatmap, opt):
    numJoints = opt.numJoints
    out = numpy.zeros((7, 256, 256))
    out[0] = scipy.misc.imresize(heatmap[0][0], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[1] = scipy.misc.imresize(heatmap[0][1], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[2] = scipy.misc.imresize(heatmap[0][2], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[3] = scipy.misc.imresize(heatmap[0][3], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[4] = scipy.misc.imresize(heatmap[0][4], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[5] = scipy.misc.imresize(heatmap[0][5], (256, 256), interp = 'bicubic', mode = 'F')-1
    out[6] = scipy.misc.imresize(heatmap[0][6], (256, 256), interp = 'bicubic', mode = 'F')-1
    joints = heatmapToJoints.heatmapToJoints(out, numJoints)
    return joints, out
