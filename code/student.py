import numpy as np
import matplotlib.pyplot as plt
from skimage import filters, feature, img_as_int
from skimage.measure import regionprops

def plot_interest_points(image, x, y):
    '''
    Plot interest points for the input image. 
    
    Show the interest points given on the input image. Be sure to add the images you make to your writeup. 

    Useful functions: Some helpful (not necessarily required) functions may include
        - matplotlib.pyplot.imshow, matplotlib.pyplot.scatter, matplotlib.pyplot.show, matplotlib.pyplot.savefig
    
    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions
    plt.imshow(image)
    plt.scatter(x, y)
    plt.show()
    

def get_feature_points(image, feature_width):
    '''
    Returns interest points for the input image

    (Please note that we recommend implementing this function last and using cheat_interest_points()
    to test your implementation of get_feature_descriptors() and match_features())

    Implement the Harris corner detector (See Szeliski 4.1.1) to start with.
    You do not need to worry about scale invariance or keypoint orientation estimation
    for your Harris corner detector.
    You can create additional interest point detector functions (e.g. MSER)
    for extra credit.

    If you're finding spurious (false/fake) interest point detections near the boundaries,
    it is safe to simply suppress the gradients / corners near the edges of
    the image.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.feature.peak_local_max (experiment with different min_distance values to get good results)
        - skimage.measure.regionprops
          
    Note: You may decide it is unnecessary to use feature_width in get_feature_points, or you may also decide to 
    use this parameter to exclude the points near image edges.

    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :feature_width: the width and height of each local feature in pixels

    :returns:
    :xs: an np array of the x coordinates of the interest points in the image
    :ys: an np array of the y coordinates of the interest points in the image

    :optional returns (may be useful for extra credit portions):
    :confidences: an np array indicating the confidence (strength) of each interest point
    :scale: an np array indicating the scale of each interest point
    :orientation: an np array indicating the orientation of each interest point

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # These are placeholders - replace with the coordinates of your interest points!

    xs = np.zeros(1)
    ys = np.zeros(1)
    # Note that xs and ys represent the coordinates of the image. Thus, xs actually denote the columns
    # of the respective points and ys denote the rows of the respective points.
    alpha = 0.01
    sigma = 1
    minimum_dist = 10
    val = -0.01

    # STEP 1: Calculate the gradient (partial derivatives on two directions).
    Ix = filters.sobel_h(image)
    Iy = filters.sobel_v(image)

    # STEP 2: Apply Gaussian filter with appropriate sigma.
    Ixx = filters.gaussian(Ix * Ix, sigma=sigma)
    Iyy = filters.gaussian(Iy * Iy, sigma=sigma)
    Ixy = filters.gaussian(Ix * Iy, sigma=sigma)

    # STEP 3: Calculate Harris cornerness score for all pixels.
    detM = (Ixx * Iyy) - (Ixy ** 2)
    trace = Ixx + Iyy
    harris_equation = detM - (alpha * (trace ** 2))

    # STEP 4: Peak local max to eliminate clusters. (Try different parameters.)
    ys = feature.peak_local_max(harris_equation, min_distance=minimum_dist, threshold_abs=val)[:, 0]
    xs = feature.peak_local_max(harris_equation, min_distance=minimum_dist, threshold_abs=val)[:, 1]
    
    # BONUS: There are some ways to improve:
    # 1. Making interest point detection multi-scaled.
    # 2. Use adaptive non-maximum suppression.

    return xs, ys


def get_feature_descriptors(image, x_array, y_array, feature_width):
    '''
    Returns features for a given set of interest points.

    To start with, you might want to simply use normalized patches as your
    local feature descriptor. This is very simple to code and works OK. However, to get
    full credit you will need to implement the more effective SIFT-like feature descriptor
    (See Szeliski 4.1.2 or the original publications at
    http://www.cs.ubc.ca/~lowe/keypoints/)

    Your implementation does not need to exactly match the SIFT reference.
    Here are the key properties your (baseline) feature descriptor should have:
    (1) a 4x4 grid of cells, each feature_width / 4 pixels square.
    (2) each cell should have a histogram of the local distribution of
        gradients in 8 orientations. Appending these histograms together will
        give you 4x4 x 8 = 128 dimensions.
    (3) Each feature should be normalized to unit length

    You do not need to perform the interpolation in which each gradient
    measurement contributes to multiple orientation bins in multiple cells
    As described in Szeliski, a single gradient measurement creates a
    weighted contribution to the 4 nearest cells and the 2 nearest
    orientation bins within each cell, for 8 total contributions. This type
    of interpolation probably will help, though.

    You do not have to explicitly compute the gradient orientation at each
    pixel (although you are free to do so). You can instead filter with
    oriented filters (e.g. a filter that responds to edges with a specific
    orientation). All of your SIFT-like features can be constructed entirely
    from filtering fairly quickly in this way.

    You do not need to do the normalize -> threshold -> normalize again
    operation as detailed in Szeliski and the SIFT paper. It can help, though.

    Another simple trick which can help is to raise each element of the final
    feature vector to some power that is less than one.

    Useful functions: A working solution does not require the use of all of these
    functions, but depending on your implementation, you may find some useful. Please
    reference the documentation for each function/library and feel free to come to hours
    or post on Piazza with any questions

        - skimage.filters (library)


    :params:
    :image: a grayscale or color image (your choice depending on your implementation)
    :x: np array of x coordinates of interest points
    :y: np array of y coordinates of interest points
    :feature_width: in pixels, is the local feature width. You can assume
                    that feature_width will be a multiple of 4 (i.e. every cell of your
                    local SIFT-like feature will have an integer width and height).
    If you want to detect and describe features at multiple scales or
    particular orientations you can add input arguments. Make sure input arguments 
    are optional or the autograder will break.

    :returns:
    :features: numpy array of computed features. It should be of size
            [num points * feature dimensionality]. For standard SIFT, `feature
            dimensionality` is 128. `num points` may be less than len(x) if
            some points are rejected, e.g., if out of bounds.

    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions
    
    # STEP 1: Calculate the gradient (partial derivatives on two directions) on all pixels.
    # STEP 2: Decompose the gradient vectors to magnitude and direction.
    # STEP 3: For each interest point, calculate the local histogram based on related 4x4 grid cells.
    #         Each cell is a square with feature_width / 4 pixels length of side.
    #         For each cell, we assign these gradient vectors corresponding to these pixels to 8 bins
    #         based on the direction (angle) of the gradient vectors. 
    # STEP 4: Now for each cell, we have a 8-dimensional vector. Appending the vectors in the 4x4 cells,
    #         we have a 128-dimensional feature.
    # STEP 5: Don't forget to normalize your feature.
    
    # BONUS: There are some ways to improve:
    # 1. Use a multi-scaled feature descriptor.
    # 2. Borrow ideas from GLOH or other type of feature descriptors.

    # This is a placeholder - replace this with your features!
    features = np.zeros((len(x_array),512))
    x_gradient = filters.sobel_h(image)
    y_gradient = filters.sobel_v(image)
    gradient_magnitude = np.sqrt(x_gradient ** 2 + y_gradient ** 2)
    gradient_origin = np.arctan2(y_gradient, x_gradient)
    
    for i in range(len(x_array)):
        features[i] = SIFT_descriptor(x_array[i], y_array[i], feature_width, gradient_origin, gradient_magnitude)

    return features

def SIFT_descriptor(x, y, feature_width, gradient_origin, gradient_magnitude):
    descriptor = []
    total = 0
    for l in range(x - feature_width // 2, x + feature_width // 2, 4):
        for k in range(y - feature_width // 2, y + feature_width // 2, 4):
            hist = np.zeros(8)
            for i in range(4):
                for j in range(4):
                    origin = gradient_origin[k + j, l + i]
                    magnitude = gradient_magnitude[k + j, l + i]
                    index = int(origin // (np.pi / 4))
                    hist[index] = hist[index] + magnitude
                    total = total + magnitude
                descriptor.append(hist)
                
    descriptor = np.array(descriptor)

    return descriptor.flatten()


def match_features(im1_features, im2_features):
    '''
    Implements the Nearest Neighbor Distance Ratio Test to assign matches between interest points
    in two images.

    Please implement the "Nearest Neighbor Distance Ratio (NNDR) Test" ,
    Equation 4.18 in Section 4.1.3 of Szeliski.

    For extra credit you can implement spatial verification of matches.

    Please assign a confidence, else the evaluation function will not work. Remember that
    the NNDR test will return a number close to 1 for feature points with similar distances.
    Think about how confidence relates to NNDR.

    This function does not need to be symmetric (e.g., it can produce
    different numbers of matches depending on the order of the arguments).

    A match is between a feature in im1_features and a feature in im2_features. We can
    represent this match as a the index of the feature in im1_features and the index
    of the feature in im2_features

    :params:
    :im1_features: an np array of features returned from get_feature_descriptors() for interest points in image1
    :im2_features: an np array of features returned from get_feature_descriptors() for interest points in image2

    :returns:
    :matches: an np array of dimension k x 2 where k is the number of matches. The first
            column is an index into im1_features and the second column is an index into im2_features
    :confidences: an np array with a real valued confidence for each match
    '''

    # TODO: Your implementation here! See block comments and the homework webpage for instructions

    # These are placeholders - replace with your matches and confidences!
    
    # STEP 1: Calculate the distances between each pairs of features between im1_features and im2_features.
    #         HINT: https://browncsci1430.github.io/webpage/hw2_featurematching/efficient_sift/
    # STEP 2: Sort and find closest features for each feature, then performs NNDR test.
    
    # BONUS: Using PCA might help the speed (but maybe not the accuracy).

    # matches = np.zeros((1,2))
    # confidences = np.zeros(1)
    F1 = np.sum(np.square(im1_features), axis=1, keepdims=True)
    F2 = np.transpose(np.sum(np.square(im2_features), axis=1, keepdims=True))
    A = F1 + F2
    B = 2 * (np.dot(im1_features, np.transpose(im2_features)))
    D = np.sqrt(A - B)

    index = np.sort(D, axis=1)
    sort = np.argsort(D, axis=1)
    nearest_neighbor = index[:, 0] / index[:, 1]
    confidences = 1 - nearest_neighbor
    feature_length = len(im1_features)
    matches = np.zeros((feature_length, 2))
    matches[:, 0] = np.arange(0, feature_length)
    matches[:, 1] = sort[:, 0]

    return matches, confidences