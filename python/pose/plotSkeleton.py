# Plot skeleton

from collections import namedtuple
import numpy
import matplotlib.pyplot

def plotSkeleton(j, opts = None, handle = None, dominantOnly = None):
    if dominantOnly == None:
        dominantOnly = False

    if not opts:
        opts = namedtuple('Options', 'clr, linewidth, jointsize')
        opts = plotSkeletonDefaultopts(opts)

    if not hasattr(opts, 'jointlinewidth'):
        optstmp = opts
        opts = namedtuple('Options', optstmp._fields + ('jointlinewidth',))
        opts.clr = optstmp.clr
        opts.linewidth = optstmp.linewidth
        opts.jointsize = optstmp.jointsize
        opts.jointlinewidth = 1
    
    if not hasattr(opts, 'jointlinecolor'):
        optstmp = opts
        opts = namedtuple('Options', optstmp._fields + ('jointlinecolor',))
        opts.clr = optstmp.clr
        opts.linewidth = optstmp.linewidth
        opts.jointsize = optstmp.jointsize
        opts.jointlinewidth = optstmp.jointlinewidth
        opts.jointlinecolor = numpy.zeros((7,3))
    
    if numpy.isscalar(opts.jointsize):
        opts.jointsize = opts.jointsize * numpy.ones((7,1))
    
    if numpy.isscalar(opts.jointlinewidth):
        opts.jointlinewidth = opts.jointlinewidth * numpy.ones((7,1))
    
    joints = numpy.arange(0,7)

    # wrist only plot
    if not handle:
        if j.shape[1] == 2:
            joints = numpy.arange(0,2)
            dontPlotSkeleton = True
        else:
            dontPlotSkeleton = False
        if j.shape[1] == 3:
            joints = numpy.arange(0,3)
            dontPlotSkeleton = True
        handle = namedtuple('handle', 'axis, ula, ura, lla, lra, joints')
        handle.axis = matplotlib.pyplot.gca()
        if not dontPlotSkeleton:
            # draw skeleton
            if not dominantOnly:
                handle.ula = matplotlib.pyplot.plot(j[0, [4,6]], j[1, [4,6]], 'y-', axes = handle.axis, linewidth = opts.linewidth, color = opts.clr[7])
            handle.ura = matplotlib.pyplot.plot(j[0, [3, 5]], j[1, [3, 5]], 'y-', axes = handle.axis, linewidth = opts.linewidth, color = opts.clr[7])
            if not dominantOnly:
                handle.lla = matplotlib.pyplot.plot(j[0, [2, 4]], j[1, [2, 4]], 'r-', axes = handle.axis, linewidth = opts.linewidth, color = opts.clr[8])
            handle.lra = matplotlib.pyplot.plot(j[0, [1, 3]], j[1, [1, 3]], 'r-', axes = handle.axis, linewidth = opts.linewidth, color = opts.clr[8])
        # draw joints
        if dominantOnly:
            joints = numpy.array([0,1,3,5])
        handlejoints = []
        for c in numpy.arange(0, 7):
            handlejoints.append(None)
        for c in joints:
            handlejoints[c] = matplotlib.pyplot.plot(j[0, c], j[1, c], 'bo', axes = handle.axis, markerfacecolor = opts.clr[c], markersize = opts.jointsize[c], linewidth = opts.jointlinewidth[c], color = opts.jointlinecolor[c])
        handle.joints = handlejoints
    else:
        # draw skelton
        setp(handle.lla, xdata = j[0, 2:5], ydata = j[1, 2:5])
        setp(handle.ula, xdata = j[0, 4:7], ydata = j[1, 4:7])
        setp(handle.lra, xdata = j[0, 1:4], ydata = j[1, 1:4])
        setp(handle.ura, xdata = j[0, 3:6], ydata = j[1, 3:6])
        for c in numpy.arange(0, 7):
            setp(handle.joints[c], xdata = j[0, c], ydata = j[1, c])
    
    return handle

def plotSkeletonDefaultopts(opts):
    opts.clr  = numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
    opts.linewidth = 2
    opts.jointsize = 6
    return opts
