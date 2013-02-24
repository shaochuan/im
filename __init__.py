'''
    Image related submodule.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''

import cv, cv2, numpy

class Font(object):
    pass

class Color(object):
    pass

font = Font()
font.default = cv.InitFont(cv.CV_FONT_HERSHEY_DUPLEX, 1, 1, thickness=2)
font.small = cv.InitFont(cv.CV_FONT_HERSHEY_DUPLEX, 0.5, 0.7, thickness=1)
font.tiny = cv.InitFont(cv.CV_FONT_HERSHEY_DUPLEX, 0.2, 0.4, thickness=1)

color = Color()
color.red = (0,0,255)
color.green = (0,255,0)
color.darkgreen = (0,128,0)
color.blue = (255,0,0)
color.cyan = (255,255,0)
color.yellow = (0,255,255)
color.purple = (255,0,255)

depth_map = {
        cv.IPL_DEPTH_8U : numpy.uint8,
        cv.IPL_DEPTH_16U : numpy.uint16
        }

def clone(iplimage, _type):
    """
        Clone the image.
        @param _type: a string, and it can be one of the following:
            "gray2bgr"
        @returns: the cloned IplImage
    """
    if _type.lower() == "gray2bgr":
        ret = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 3)
        r = cv.CloneImage(iplimage)
        g = cv.CloneImage(iplimage)
        b = cv.CloneImage(iplimage)
        cv.Merge(r,g,b,None,ret)
        return ret
    else:
        raise ValueError("Unknown _type value.")

def paste(iplimage, canvas, x, y):
    """
        Paste the image. Canvas image will be modified.
        @param iplimage: the image to be pasted.
        @param canvas: the canvas to be pasted to.
        @x, y: the coordinate to be pasted at.
        @returns: None
    """
    cv.SetImageROI(canvas, (x, y, x+iplimage.width, y+iplimage.height))
    cv.Copy(iplimage, canvas)
    cv.ResetImageROI(canvas)

def stitch_stacking(iplimg1, iplimg2):
    assert iplimg1.depth == iplimg2.depth
    assert iplimg2.channels == iplimg2.channels
    total_width = max(iplimg1.width, iplimg2.width)
    total_height = iplimg1.height + iplimg2.height
    size = (total_width, total_height)
    iplimage = cv.CreateImage(size, iplimg1.depth, iplimg1.channels)
    paste(iplimg1, iplimage, 0, 0)
    paste(iplimg2, iplimage, 0, iplimg1.height)
    return iplimage

def newgray(cvimg_or_size):
    """
        Create a new iplimage with single channel.
        @param cvimg_or_size: an iplimage or a tuple representing the size.
        @returns: IplImage.
    """
    size = None
    if isinstance(cvimg_or_size, (tuple, list)):
        size = tuple(cvimg_or_size)
    else:
        size = cv.GetSize(cvimg)
    return cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)

def resize(iplimage, newsize):
    """
        Create a new IplImage with the new image size.
        @param ipliamge: the original IplImage.
        @param newsize: the new image size.
        @returns: a new IplImage.
    """
    size = cv.GetSize(iplimage)
    newimg = cv.CreateImage(newsize, iplimage.depth, iplimage.channels)
    cv.Resize(iplimage, newimg)
    return newimg

def split3(iplimage):
    if iplimage.nChannels != 3:
       raise ValueError("The channel is not consistent.")
    b = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    g = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    r = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    cv.Split(iplimage, b, g, r, None)
    return b,g,r

def split4(iplimage):
    if iplimage.nChannels != 4:
        raise ValueError("The channel is not consistent.")
    b = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    g = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    r = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    a = cv.CreateImage((iplimage.width, iplimage.height), cv.IPL_DEPTH_8U, 1)
    cv.Split(iplimage, b, g, r, a)
    return b,g,r,a

def to_npimage(iplimage):
    width, height = cv.GetSize(iplimage)
    shape = (height, width, iplimage.channels)
    nparray = numpy.fromstring(iplimage.tostring(),
                               dtype=depth_map[iplimage.depth])
    npimage = nparray.reshape(shape)
    return npimage

def imgray(iplimage):
    npgray = cv2.cvtColor(to_npimage(iplimage),
                          cv.CV_RGB2GRAY)
    return cv.GetImage(cv.fromarray(npgray))

def drawtext(img, text, x, y, font=font.default, color=color.red):
    cv.PutText(img, text, (x,y), font, color)

def center_of_mass(contour):
    moment = cv.Moments(contour)
    mass = cv.GetSpatialMoment(moment, 0, 0)
    mx = cv.GetSpatialMoment(moment,1,0)
    my = cv.GetSpatialMoment(moment,0,1)
    X = mx/mass
    Y = my/mass
    return X,Y

def find_contour(bwimg):
    storage = cv.CreateMemStorage(0)
    contour = cv.FindContours(bwimg,
                    storage,cv.CV_RETR_LIST,
                    cv.CV_CHAIN_APPROX_SIMPLE)
    return contour

def extract_sift(iplimage):
    gray = iplimage
    if iplimage.channels > 1:
        gray = imgray(iplimage)
    assert gray.channels == 1, "expecting 'gray' to be single channel image."
    npgray = to_npimage(gray)
    sift = cv2.SIFT()#contrastThreshold=0.1, edgeThreshold=50)
    keypoints = sift.detect(npgray)
    desc = cv2.DescriptorExtractor_create('SIFT')
    descriptors = desc.compute(npgray, keypoints)
    return descriptors

