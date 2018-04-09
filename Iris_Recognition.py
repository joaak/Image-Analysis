import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

def iris_localization(eye):

    # binarizing the image to get the coordinates of the pupil and remove some eyelash noise
    binarized_image = eye.copy()
    for i in range(binarized_image.shape[0]):
        for j in range(binarized_image.shape[1]):
            if binarized_image[i][j] < 65:
                binarized_image[i][j] = 0
            else:
                if j < 75 or j > 250:
                    binarized_image[i][j] = 0
                else:
                    binarized_image[i][j] = 1

    # horizontal and vertical projection of the binarized image to get the coordinates of the centre of the pupil
    (rows,cols) = eye.shape
    h_projection_1 = np.array([x/255/rows for x in binarized_image.sum(axis=0)])
    v_projection_1 = np.array([x/255/cols for x in binarized_image.sum(axis=1)])
    approximate_centroid_1 = ( np.argmin(h_projection_1), np.argmin(v_projection_1))

    # horizontal and vertical projection of the original image to get the coordinates of the centre of the pupil
    h_projection_2 = np.array([x/255/rows for x in eye.sum(axis=0)])
    v_projection_2 = np.array([x/255/cols for x in eye.sum(axis=1)])
    approximate_centroid_2 = (np.argmin(h_projection_2), np.argmin(v_projection_2))

    # since both projections are prone to errors (eyelash noise etc.) we choose to take the maximum of the two estimates
    Xp = max(approximate_centroid_1[0], approximate_centroid_2[0])
    Yp = max(approximate_centroid_1[1], approximate_centroid_2[1])
    
    # now we take a 160 x 160 frame centered around the pupil to reduce the image size and speed up computation
    crop_eye = eye[Yp-80:Yp+80, Xp-80:Xp+80]

    # create a copy of the cropped eye to draw the pupil circle on (for sanity check during execution)
    crop_eye_copy = crop_eye.copy()

    # we run hough circles on the original image to get an estimate of the radius of the pupil circle
    # set parameters in a way to ensure only one most prominent circle is detected in the image
    pupil_circles = cv2.HoughCircles(crop_eye_copy, cv2.HOUGH_GRADIENT, 5.1, 300, minRadius = 40, maxRadius=50, param2=50)

    # draw pupil circle on the image
    if pupil_circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        pupil_circles = np.round(pupil_circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in pupil_circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(crop_eye_copy, (x, y), r, color=(0, 255, 0), thickness=3)

    # create a copy of the eye image for future changes 
    eye_copy = eye.copy()

    # run canny edge detection on the original image
    eye_edges = cv2.Canny(eye_copy, threshold1 = 50, threshold2 = 70)

    # the hough circles algorithm returns both types of array formats, so we ensure that we don't get an error because of that
    try:
        Rp = int(pupil_circles[0][2])
    except:
        Rp = int(pupil_circles[0][0][2])

    # remove the eyelash, pupil and other noise in the canny edge detection image using the information from the cioordinates and radius of the pupil
    eye_edges[Yp-(Rp+30):Yp+(Rp+30), Xp-(Rp+30):Xp+(Rp+30)] = 1
    for i in range(eye_edges.shape[0]):
        for j in range(eye_edges.shape[1]):
            if i < Yp-(Rp+80) or i > Yp+(Rp+80):
                eye_edges[i][j] = 1
            if j < Xp-(Rp+70) or j > Xp+(Rp+70):
                 eye_edges[i][j] = 1

    # run hough circle on the new less noisy canny edge image to get the location of the iris in the image
    iris_circles = cv2.HoughCircles(eye_edges, cv2.HOUGH_GRADIENT, 6.2, 300, minRadius=90, maxRadius=Rp+70, param2=250)

    # plot the iris circle on the copy of the eye image (for sanity check during execution)
    if iris_circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        iris_circles = np.round(iris_circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in iris_circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(eye_copy, (x, y), r, color=(0, 255, 0), thickness=3)

    # print eye_copy with the iris circle drawn to see how the localization is doing
    # print(iris_circles)

    # save the coordinates and radius of the iris  
    (Xi, Yi, Ri) = iris_circles[0]

    # return the coordinate and radius information of the iris and the pupil
    return(Xp, Yp, Rp, Xi, Yi, Ri)

# divide the area between the iris and the pupil into 64 x 512 parts and then convert the circular image into a normalized rectangle using the procedure mentioned in Li Ma's paper.
def image_normalization(eye):
    Xp, Yp, Rp, Xi, Yi, Ri = iris_localization(eye)
    norm_eye = np.zeros((64, 512))
    theta = (2*math.pi)/512
    V = (Ri - Rp)/64
    for i in range(norm_eye.shape[0]):
        R = Rp + V*(i+1)
        for j in range(norm_eye.shape[1]):
                x = int(np.round(Xp + R*np.cos(theta*j), decimals=0))
                y = int(np.round(Yp + R*np.sin(theta*j), decimals=0))
                if y < 280 and x < 320:
                    norm_eye[i][j] = eye[y][x]
                elif y > 280 and x < 320:
                    norm_eye[i][j] = eye[279][x]
                elif y < 280 and x > 320:
                    norm_eye[i][j] = eye[y][319]
                else:
                    norm_eye[i][j] = eye[279][319]
    return norm_eye

# for each 32 x 32 block of the image, carry out histogram equalization, to enhance the image
def image_enhancement(eye):
    norm_eye = image_normalization(eye)
    i = 0
    while i < norm_eye.shape[0]:
        j = 0
        while j < norm_eye.shape[1]:
            roi = norm_eye[i:i+32, j:j+32]
            roi = roi.astype('uint8')
            dest = cv2.equalizeHist(roi)
            norm_eye[i:i+32, j:j+32] = dest
            j = j+32
        i = i+32

    norm_eye = norm_eye[0:48]
    return norm_eye

# define the feature extraction functions as given in Li Ma's paper

def m(x, y, f):
    val = np.cos(2*np.pi*f*math.sqrt(x ** 2 + y ** 2))
    return val

def gabor(x, y, dx, dy, f):
    gb = (1/(2*math.pi*dx*dy)) * np.exp(-0.5 * (x ** 2 / dx ** 2 + y ** 2 / dy ** 2)) * m(x, y, f)
    return gb

# write a function to derive features from the image

def get_feature(norm_eye):

    # since Li Ma used two channel spatial filters, we have to obtain two filtered images
    dx1 = 3
    dx2 = 4.5
    dy = 1.5
    f = 1/dy

    feature1 = np.zeros_like(norm_eye)
    feature2 = np.zeros_like(norm_eye)

    # the process is to set a kernel window of 8 x 8, get a Gabor filter kernel using the functions defined and then convolve the image with that kernel
    i = 4
    while i < (norm_eye.shape[0]-4):
        j = 4
        while j < (norm_eye.shape[1]-4):
            height = 4
            width = 4
            kernel1 = np.zeros((height*2, width*2))
            kernel2 = np.zeros((height*2, width*2))
            for y in range(-height, height):
                for x in range(-width, width):
                    kernel1[x+width][y+height] = gabor(height+y+1, width+x+1, dx1, dy, f)
                    kernel2[x+width][y+height] = gabor(height+y+1, width+x+1, dx2, dy, f)
            norm_eye_block = norm_eye[i-height:i+height, j-width:j+width]
            # convolution
            v1 = cv2.filter2D(src=norm_eye_block, kernel=kernel1, ddepth=-1)
            feature1[i-height:i+height, j-width:j+width] = v1
            v2 = cv2.filter2D(src=norm_eye_block, kernel=kernel2, ddepth=-1)
            feature2[i-height:i+height, j-width:j+width] = v2
            j = j+1
        i = i+1

    # after obtaining the two filtered images, we convert them into a feature vector using the mean and standard deviation of each 8 x 8 block
    vector = []
    i = 0
    while i < feature1.shape[0]:
        j = 0
        while j < feature1.shape[1]:
            block1 = feature1[i:i+8, j:j+8]
            block2 = feature2[i:i+8, j:j+8]
            m1 = block1.mean()
            m2 = block2.mean()
            vector.append(m1)
            vector.append(m2)
            s1 = block1.std()
            s2 = block2.std()
            vector.append(s1)
            vector.append(s2)
            j = j + 8
        i = i + 8
    return vector

    # the length of this vector is 1536

# after getting the training and test data, we need to reduce their dimensionality
# we can do that by using fisher linear discriminant, which is a well-suited algorithm for this problem

def iris_matching(train_df, test,_df):
    y_train = train_df[1536]
    X_train = train_df.drop(1536, axis=1)
    y_test = test_df[1536]
    X_test = test_df.drop(1536, axis=1)

    # two types of metrics will be used to calculate the distance - cosine similarity and manhattan distance
    correct_match_score_cosine = []
    correct_match_score_manhattan = []

    # cosine similarity
    for i in tqdm(range(1,108)):
        lda = LinearDiscriminantAnalysis(n_components=i, solver='eigen', shrinkage='auto')
        lda.fit(X_train, y_train)
        transformed_X_train = lda.transform(X_train)
        transformed_X_test = lda.transform(X_test)
        # k nearest neighbors will be used to get the predicted labels
        knn_cosine = KNeighborsClassifier(metric='cosine')
        knn_cosine.fit(transformed_X_train, y_train)
        y_pred_knn_cosine = knn_cosine.predict(transformed_X_test)
        count = 0
        for j in range(len(y_test)):
            if y_test[j] == y_pred_knn_cosine[j]:
                count = count + 1
        correct_match_score_cosine.append(count/len(y_test))

    max_val_cosine = max(np.array(correct_match_score_cosine))
    max_val_feat_cosine = np.argmax(np.array(correct_match_score_cosine))

    print("The maximum accuracy of this model is ", max_val_cosine, " at ", max_val_feat_cosine, " features.")

    # manhattan distance
    for i in tqdm(range(1,108)):
        lda = LinearDiscriminantAnalysis(n_components=i, solver='eigen', shrinkage='auto')
        lda.fit(X_train, y_train)
        transformed_X_train = lda.transform(X_train)
        transformed_X_test = lda.transform(X_test)
        knn_manhattan = KNeighborsClassifier(metric='manhattan')
        knn_manhattan.fit(transformed_X_train, y_train)
        y_pred_knn_manhattan = knn_manhattan.predict(transformed_X_test)
        count = 0
        for j in range(len(y_test)):
            if y_test[j] == y_pred_knn_manhattan[j]:
                count = count + 1
        correct_match_score_manhattan.append(count/len(y_test))

    max_val_manhattan = max(np.array(correct_match_score_manhattan))
    max_val_feat_manhattan = np.argmax(np.array(correct_match_score_manhattan))

    print("The maximum accuracy of this model is ", max_val_manhattan, " at ", max_val_feat_manhattan, " features.")

    return(correct_match_score_cosine, correct_match_score_manhattan)

def performance_evaluation(correct_match_score_cosine, correct_match_score_manhattan):

    plt.plot(correct_match_score_cosine)
    plt.xlabel('Number of Features in Cosine Similarity')
    plt.ylabel('Correct Recognition Rate')
    plt.show()

    plt.plot(correct_match_score_manhattan)
    plt.xlabel('Number of Features in Manhattan Similarity')
    plt.ylabel('Correct Recognition Rate')
    plt.show()


# Main
# loop through the source folder and iterate over each image one by one, for both the test and the training data

from tqdm import tqdm
train_df = pd.DataFrame()
for i in tqdm(range(1, 109)):
    for j in range(1, 4):
        if i < 10:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/00'+str(i)+'/1/00'+str(i)+'_1_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        elif i >= 10 and i < 100:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/0'+str(i)+'/1/0'+str(i)+'_1_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        else:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/'+str(i)+'/1/'+str(i)+'_1_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        norm_eye = image_enhancement(eye)
        vector = get_feature(norm_eye)
        # append the label to the vector and then append the vector to the training dataframe
        vector.append(i)
        row=pd.Series(vector)
        train_df = train_df.append([row],ignore_index=True)

test_df = pd.DataFrame()
for i in tqdm(range(1, 109)):
    for j in range(1, 5):
        if i < 10:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/00'+str(i)+'/2/00'+str(i)+'_2_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        elif i >= 10 and i < 100:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/0'+str(i)+'/2/0'+str(i)+'_2_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        else:
            url = '/Users/aakankshajoshi/Documents/MSDS Spring 2018/Image Analysis/HW2/CASIA Iris Image/'+str(i)+'/2/'+str(i)+'_2_'+str(j)+'.bmp'
            eye = cv2.imread(url)
        eye = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)
        norm_eye = image_enhancement(eye)
        vector = get_feature(norm_eye)
        # append the label to the vector and then append the vector to the test dataframe
        vector.append(i)
        row=pd.Series(vector)
        test_df = test_df.append([row],ignore_index=True)

correct_match_score_cosine, correct_match_score_manhattan = iris_matching(train_df, test_df)
performance_evaluation(correct_match_score_cosine, correct_match_score_manhattan)


# cosine similarity correct recognition rate: 62.27%
# manhattan distance correct recognition rate: 61.34%
