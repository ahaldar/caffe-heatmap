# Caffe-heatmap

This is a fork of Caffe that enables training of heatmap regressor ConvNets for the general problem of regressing (x,y) positions in images.   
Multiple implementations are provided, using the PyCaffe interface for Python3, the MatCaffe interface for Matlab, and the standard C++ Caffe API.


## Pretrained models
- [Fusion model trained on FLIC](http://tomas.pfister.fi/models/caffe-heatmap-flic.caffemodel)
- [Fusion model trained on ChaLearn](http://tomas.pfister.fi/models/caffe-heatmap-chalearn.caffemodel)

## Pre-cropped images and training labels for FLIC
- [Training](http://tomas.pfister.fi/flic_train_cropped_multfact282.tgz)
- [Testing](http://tomas.pfister.fi/flic_test_cropped_multfact282.tgz)
- Note these files require multfact=282 in both training and testing data layers

## Testing instructions

For example codes for running the FLIC model on a video, use the following:
demo.m (on Matlab version 2015 or earlier)   
python python/pose/demo.py   
./build/examples/cpp_heatmap/demo.bin (generated automatically on running "make")   

## Paper
Please cite our [ICCV'15 paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Pfister15a/pfister15a.pdf) in your publications if this code helps your research:

      @InProceedings{Pfister15a,
        author       = "Pfister, T. and Charles, J. and Zisserman, A.",
        title        = "Flowing ConvNets for Human Pose Estimation in Videos",
        booktitle    = "IEEE International Conference on Computer Vision",
        year         = "2015",
      }


# Caffe

Caffe is a deep learning framework made with expression, speed, and modularity in mind.
It is developed by the Berkeley Vision and Learning Center ([BVLC](http://bvlc.eecs.berkeley.edu)) and community contributors.

Check out the [project site](http://caffe.berkeleyvision.org) for all the details like

- [DIY Deep Learning for Vision with Caffe](https://docs.google.com/presentation/d/1UeKXVgRvvxg9OUdh_UiC5G71UMscNPlvArsWER41PsU/edit#slide=id.p)
- [Tutorial Documentation](http://caffe.berkeleyvision.org/tutorial/)
- [BVLC reference models](http://caffe.berkeleyvision.org/model_zoo.html) and the [community model zoo](https://github.com/BVLC/caffe/wiki/Model-Zoo)
- [Installation instructions](http://caffe.berkeleyvision.org/installation.html)

and step-by-step examples.

[![Join the chat at https://gitter.im/BVLC/caffe](https://badges.gitter.im/Join%20Chat.svg)](https://gitter.im/BVLC/caffe?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

Please join the [caffe-users group](https://groups.google.com/forum/#!forum/caffe-users) or [gitter chat](https://gitter.im/BVLC/caffe) to ask questions and talk about methods and models.
Framework development discussions and thorough bug reports are collected on [Issues](https://github.com/BVLC/caffe/issues).

Happy brewing!

## License and Citation

Caffe is released under the [BSD 2-Clause license](https://github.com/BVLC/caffe/blob/master/LICENSE).
The BVLC reference models are released for unrestricted use.

Please cite Caffe in your publications if it helps your research:

    @article{jia2014caffe,
      Author = {Jia, Yangqing and Shelhamer, Evan and Donahue, Jeff and Karayev, Sergey and Long, Jonathan and Girshick, Ross and Guadarrama, Sergio and Darrell, Trevor},
      Journal = {arXiv preprint arXiv:1408.5093},
      Title = {Caffe: Convolutional Architecture for Fast Feature Embedding},
      Year = {2014}
    }

