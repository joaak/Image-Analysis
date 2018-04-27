# Image-Analysis: Exploring Image Analysis and its Applications 

## Face Detection

Detect a face in an image using k-means clustering based on color. 

## Iris Recognition

Implementation of John Daugman's Iris Detection algorithm and Li Ma's spatial filters to detect and recognise iris. 

### 1. Iris Localization

Obtain the exact parameters of the pupil and iris circles using Canny edge detection and Hough transform in a certain region determined by the center of the pupil.

Return the coordinate and radius information for both iris and pupil.

### 2. Iris Normalization

Map the iris from Cartesian coordinates to polar coordinates using the method provided in Li Ma's paper. Return a normalized image with dimensions 68 x 512.


### 3. Image Enhancement

The normalized iris image has low contrast and may have nonuniform brightness caused by the position of light sources. We enhance the image by means of histogram equalization in each 32 x 32 region of the image. Li Ma's Region of Interest consists of only 48 x 512 part of the image so we return the enhanced image of only this ROI.


### 4. Feature Extraction

Gabor filters have been used to filter the image and 2D convolution has been used to convolve the original image with the filter to obtain the features in the image. To characterize local texture information of the iris, statistical features in each 8 x 8 small block of the two filtered images have been extracted. In total we have 768 blocks and for each small block, two feature values are captured. This generates 1,536 feature components. The feature values used in the algorithm are the mean and the average absolute deviation of the magnitude of each filtered block.

### 5. Iris Matching

## Brain State Classification

1. Using the MATLAB sofware SPM to identify the brain regions that repond to auditory signals based on brain images. 
2. Using Python for brain state classification challenges. 

Linear Discriminant Analysis is used to reduce the dimensionality of the high-dimensionality dataset. Eigen value decomposition is used as the solver, the shrinkage is automatic and the number of components has been seen in a range from 1 to the 108.

After getting the reduced dataset, the model is trained using k nearest neighbors where the metrics being compared are 'cosine similarity' and 'manhattan distance'.

The knn model learns from the training set and makes predictions based on the information it learns. The accuracy of the model is calculated in terms of the correct recognition rate (how many matches were correct), and this value is ~62.27% for cosine similarity and ~61.34% for Manhattan distance. A check was also performed to see that the values of the model's performances converge at these values.


### 6. Limitations

There is still some imprecision as the eyelashes and the image noise have not been completely dealt with.


