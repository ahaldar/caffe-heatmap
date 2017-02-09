# Wrapper to run network on multiple images

import numpy

import initCaffe
import applyNetImage

def applyNet(files, opt):
    print("config:\n")
    print(opt.visualise)
    print(opt.useGPU)
    print(opt.dims)
    print(opt.numJoints)
    print(opt.layerName)
    print(opt.modelDefFile)
    print(opt.modelFile)
    print(opt.inputDir)
    print(opt.numFiles)
    print("\n")

    # Initialise caffe
    net = initCaffe.initCaffe(opt)

    # Apply network separately to each image
    joints = numpy.zeros((2, opt.numJoints, opt.numFiles))
    for ind in numpy.arange(0, opt.numFiles):
        imFile = files[ind]
        print("file: %s\n" % imFile)
        joints[:, :, ind] = applyNetImage.applyNetImage(imFile, net, opt)
        if opt.visualise:
            input("Press ENTER to continue.")            
    return joints
