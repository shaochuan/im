'''
    Image related submodule.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''

import cv


def clone(src, _type):
    """
        Clone the image.
        @param _type: a string, and it can be one of the following:
            "gray2bgr"
        @returns: the cloned image
    """
    if _type.lower() == "gray2bgr":
        ret = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 3)
        r = cv.CloneImage(src)
        g = cv.CloneImage(src)
        b = cv.CloneImage(src)
        cv.Merge(r,g,b,None,ret)
        return ret
    else:
        raise ValueError("Unknown _type value.")

def paste(img, canvas, x, y):
    cv.SetImageROI(canvas, (x,y,x+img.width,y+img.height))
    cv.Copy(img, canvas)
    cv.ResetImageROI(canvas)

def newgray(cvimg_or_size):
    size = None
    if isinstance(cvimg_or_size, (tuple, list)):
        size = tuple(cvimg_or_size)
    else:
        size = (cvimg_or_size.width, cvimg_or_size.height)
    return cv.CreateImage(size, cv.IPL_DEPTH_8U, 1)

def split3(src):
    if src.nChannels != 3:
       raise ValueError("The channel is not consistent.")
    b = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    g = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    r = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    cv.Split(src, b, g, r, None)
    return b,g,r

def split4(src):
    if src.nChannels != 4:
        raise ValueError("The channel is not consistent.")
    b = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    g = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    r = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    a = cv.CreateImage((src.width, src.height), cv.IPL_DEPTH_8U, 1)
    cv.Split(src, b, g, r, a)
    return b,g,r,a


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
