# Initialise Caffe

import caffe

def initCaffe(opt):
    caffe.set_mode_gpu()
    gpu_id = 0
    caffe.set_device(gpu_id)
    net = caffe.Net(opt.modelDefFile, opt.modelFile, caffe.TEST)
    return net
