'''
    Image related submodule.

    @date: Mar. 29, 2011
    @author: Shao-Chuan Wang (sw2644 at columbia.edu)
'''

import cv

red = (0,0,255)
green = (0,255,0)
blue = (255,0,0)

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


