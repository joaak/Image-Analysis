## Preprocessing Steps

### 1. Realign

The goal is to correct any motion disturbances present during the fMRI scan. The mean image of all time points is taken as the reference time volume. The translation and rotation graphs can be seen in Figure 30.2.

### 2. Co-register

The alignment of anatomical scans and functional scans. the reference image is the mean functional image and the source image is the T1 weighted image which is co-registered with the reference image.

### 3. Segment

The for each subject are classified into a number of different tissue types. The tissue types are defined according to tissue probability maps and the tissue types include grey matter, white matter, skull and the cerebralspinal fluid.

### 4. Normalisation

The goal is to fit the image into a standard template brain.

### 5. Smoothing

The burring of the image done by convolution in the spatial domain (SPM uses Gaussian kernel). Smoothing is done to make data more normally distributed and increase validity of statistical testing.

## Drawbacks

More parameters could have been tested to check for better results.

## Result

From all the different fMRI images we can draw conclusions about the parts of brain which respond to auditory signals.
