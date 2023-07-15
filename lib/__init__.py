import numpy as np

def find_nearest_index(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class FigureSize():
    THIN     = [24, 3]
    NARROW   = [24, 8]
    MEDIUM   = [24, 15]
    LARGE    = [24, 24]