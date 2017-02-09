# Visualise heatmap

import numpy
from skimage import color
from skimage import filter

def getConfidenceImage(dist, segcpimg_crop, clr = None):
    num_points = dist.shape[0]
    numpy.asarray(dist, dtype = numpy.float)
    m, n, _ = segcpimg_crop.shape
    bbox = numpy.array([1, 1, n, m])
 
    clrstr = numpy.array([[0, 1, 0], [0, 0, 1], [1, 0, 0], [1, 0, 1], [0, 1, 1], [1, 1, 0], [0, 0, 0]])

    if clr != None:
        clrstr = numpy.tile(clr, (9, 1))

    uintimg = numpy.asarray(segcpimg_crop, dtype = numpy.uint8)
    grayimg = color.rgb2gray(uintimg);
    edges = filter.canny(grayimg)
    background = numpy.asarray(~edges, dtype = numpy.float)
    bg = background[:, :, numpy.newaxis]
    pdf_img = 0.2 * numpy.tile(bg, (1, 1, 3)) + 0.8 * numpy.ones((bbox[3], bbox[2], 3))

    # Normalise the distributions for visualisation
    for c in numpy.arange(num_points-1, -1, -1):
        d = dist[c, :, :] / numpy.amax(numpy.amax(dist[c, :, :]))
        dd = d[:, :, numpy.newaxis]
        alpha = numpy.tile(dd, (1,1,3))
        t = clrstr[c, :, numpy.newaxis, numpy.newaxis]
        single_joint_pdf = numpy.tile(t, (1, dist.shape[1], dist.shape[2]))
        single_joint_pdf = numpy.transpose(single_joint_pdf, (1, 2, 0))
        pdf_img = alpha * single_joint_pdf + (1 - alpha) * pdf_img
                    
    return pdf_img
