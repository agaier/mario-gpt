import numpy as np

def rgb_to_gray(image):
    """
    Convert an RGB image to grayscale using the luminance formula.
    
    The function uses the following formula to calculate the grayscale values:
    L = 0.2989 * R + 0.5870 * G + 0.1140 * B
    
    Parameters
    ----------
    image : numpy.ndarray
        A 3D array representing the RGB image with dimensions [width, height, rgb].
        
    Returns
    -------
    numpy.ndarray
        A 2D array representing the grayscale image with dimensions [width, height].
        
    Raises
    ------
    ValueError
        If the input image is not a 3D array of dimensions [width, height, rgb].
    """
    if len(image.shape) == 3 and image.shape[2] == 3:
        # use the luminance formula to convert the RGB image to grayscale
        gray_image = 0.2989 * image[:,:,0] + 0.5870 * image[:,:,1] + 0.1140 * image[:,:,2]
        # normalize the grayscale values to the range [0, 255]
        gray_image = (gray_image / np.max(gray_image) * 255).astype(np.uint8)
        return gray_image
    else:
        raise ValueError('Input image must be a 3D array of dimensions [width, height, rgb]')


def map_array_to_colors(array):
    h, w = array.shape
    colors = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            colors[i, j] = map_to_color(array[i, j])
    return colors


def map_to_color(char):
    # colors = {
    #     'B': [0, 170, 0],
    #     'C': [185, 122, 87],
    #     'D': [255, 242, 0],
    #     'E': [191, 232, 242]
    # }
    colors = {
        'A': [0, 170, 0],
        'B': [185, 122, 87],
        'C': [255, 242, 0],
        'D': [191, 232, 242]
    }
    return colors.get(char, [255, 0, 0])  

def count_flowers(array):
    """
    Count the number of flowers in a given 2D or 3D array.
    
    The function first converts the input array to grayscale if necessary.
    It then iterates through the array, checking for a 3x3 flower pattern.
    The flower pattern is defined as:
    
    [[255, 252, 255],
     [252, 115, 252],
     [255, 252, 255]]
    
    Parameters
    ----------
    array : numpy.ndarray
        A 2D or 3D array representing the pixel art.
        
    Returns
    -------
    int
        The number of flowers found in the input array.
    """
    gray_array = rgb_to_gray(array)    
    flower_pattern = np.array([[255, 252, 255],
                               [252, 115, 252],
                               [255, 252, 255]])
    flower_count = 0
    for i in range(gray_array.shape[0] - 2):
        for j in range(gray_array.shape[1] - 2):
            if np.array_equal(gray_array[i:i+3, j:j+3], flower_pattern):
                flower_count += 1                
    return flower_count


def calculate_crookedness_score(array, stem_value=115):
    """
    Calculate the crookedness score of stems in a 2D pixel art array.
    
    The function iterates through each 3x3 patch in the input array and
    increases the crookedness score by 1 for each different y value of a
    stem pixel in a patch. A perfectly straight stem patch will add 0 to
    the score, while a patch that is moving at a diagonal will add 2.
    
    Parameters
    ----------
    array : numpy.ndarray
        A 2D array representing the pixel art.
    stem_value : int, optional
        The value representing the stem pixels, by default 115.
        
    Returns
    -------
    int
        The crookedness score of stems in the input array.

    TODO
    ----
    - Normalize to the number of stems in the array

    """
    # convert the input array to grayscale if necessary
    if len(array.shape)==3:
        array = rgb_to_gray(array)
        
    crookedness_score = 0
    for i in range(array.shape[0] - 2):
        for j in range(array.shape[1] - 2):
            patch = array[i:i+3, j:j+3]
            stem_positions = np.argwhere(patch == stem_value)
            
            if stem_positions.size > 0:
                unique_y_values = np.unique(stem_positions[:, 1])
                crookedness_score += len(unique_y_values) - 1
                
    return crookedness_score
