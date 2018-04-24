import numpy as np
from skimage.feature import local_binary_pattern as lbp

def LBP(image, points=8, radius=1):
    '''
    Uniform Local Binary Patterns algorithm
    Input image with shape (height, width, channels)
    Output features with length 59 * number of channels
    '''
    # lbp for all channels of image
    histogram = np.empty(0, dtype=np.int)
    for i in range(image.shape[2]):
        channel = image[:, :, i]
        pattern = lbp(channel, points, radius, method='nri_uniform')
        pattern = pattern.astype(np.int).ravel()
        pattern = np.bincount(pattern)
        if len(pattern) < 59:
            pattern = np.concatenate((pattern, np.zeros(59 - len(pattern))))
        histogram = np.concatenate((histogram, pattern))
    # normalize the histogram and return it
    features = (histogram - np.mean(histogram)) / np.std(histogram)
    return features
    

    
