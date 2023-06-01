import numpy as np
from skimage import exposure
import matplotlib.pyplot as plt
from matplotlib import rcParams
from astropy.visualization import ImageNormalize as IN



def ImageNormalise(images,vmin,vmax):
    
    images = np.squeeze(images)
    #print(vmax)
    images = IN(data = images,vmin = vmin, vmax =vmax,clip =True)
    return images