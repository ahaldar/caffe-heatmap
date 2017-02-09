# Prepare input image for caffe: change to single & permute color channels

import numpy

def prepareImagePose(img, opt):
    img = img.astype('float32')
    imgOut = numpy.transpose(img, (2, 0, 1))
    return imgOut

    
