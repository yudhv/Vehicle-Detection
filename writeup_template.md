##Vehicle Detection Project

The steps followed to identify cars on the road were as follows:

* Download and extract "car" and "non-car" images from [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [here](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) respectively. 
* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images 
* Append color and histogram features to the HOG feature vector. 
* Train a Linear SVM classifier to distinguish between vehicle and non-vehicle images using the aforementioned feature vector extraction
* Implement a sliding-window technique and use the trained classifier to search for vehicles in said window.
* Run the above steps in a pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/car_hog.jpg
[image8]: ./examples/noncar_hog.jpg
[image3]: ./examples/aggr_win1.jpg
[image4]: ./examples/aggr_win2.jpg
[image5]: ./examples/heat.jpg
[image6]: ./examples/labels.jpg
[image7]: ./examples/final.jpg
[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  

--- 
###Training Data

The code for this step can be found in the 13th code cell of the IPython notebook "main.ipynb" which will be referenced as simply the notebook from now on.

Only the first 500 images from both car and the non-car sets were used for the project as the non-car images after this point became very repetitive and lead to an artificial testing accuracy. This in turn would generally lead to near zero positive identifications of cars on the road.

Once extracted, the images are passed to an `extract_features()` function that combines HOG, color and histogram feature extractions into a single feature vector and returns it. This feature vector is then used to train a Support Vector Machine Classifier.

Here is an example of the car and non-car images - 

![alt text][image1]

###Histogram of Oriented Gradients (HOG)

The code for this step is contained in the third code cell as the function `get_hog_feature()` and the function `extract_features()` in the 7th code cell of the notebook. 

HOG stands for Histogram of Oriented Gradients. It is used to capture the shape of structures in an image by capturing information about gradients. In this project, I use the `hog()` function in the skimage library to compute the said gradients. This function is used for both the training images, as well as the actual video frames. It takes as input the following - 
1. `orients` tells the number of gradients to divide the features into (i.e the number of directions to compute)
2. `pixels_per_cell` is a 2-tuple that tells the cell size over which each gradient histogram must be computed. 
3. `cells_per_block` is also a 2-tuple that defines the number of cells to put in each block. Blocks are required to make more robust gradient computations, but are completely optional. They specify the local area over which the histogram counts in a given cell will be normalized.

I tried various combinations of parameters and found the following values to perform the best
1. `orients = 9` - The car images from the final video simply didn't have enough resolution to justify a high number of gradients (as a higher resolution would've defined clear cut lines and slopes. A smaller value for the `orients` parameter in turn helped speed up all processes significantly.
2. `pixels_per_cell = 8` - Each gradient getting 8 pixels helped in keeping the gradients locally optimized by capturing a good amount of details in 1280 * 720 video images. A lower number was slowing down the process, and a higher number simply didn't capture accurate information which showed in the higher number of false positives like trees, silver-colored poles, etc. 
3. `cells_per_block = 2` - This meant the averaging of gradients at a 16 * 16 resolution, which felt right intuitively. Going too high could make the Classifier too generalized and would've let it more false positives.

Here is the resultant HOG image -

![alt text][image2]

For a non_car image - 

![alt text][image8]

###Spatial and Color Histograms

The code for this step is contained in the 5th and 6th code cell as the functions `bin_spatial()` and `color_hist()` respectively.

An additional two feature sets were used to augment the HOG feature extraction mentioned above. 

The `bin_spatial()` function simply downscales the three layers of an image and linearizes them to form a feature vector.

The `color_hist()` function computes a histogram of the three layers of an image an concatenates those values into the third and final feature vector. 

Finally, these three feature vectors (HOG, Spatial, and Histogram) are stacked together horizontally and fed to the Support Vector Classifier for training/predicting.

###Support Vector Classifier (SVC)

The code for this step can be found in the 13th code cell of the notebook.

Here, I use a Linear Support Vector Machine Classifier (SVC in short) to learn the difference between features arising from car images and features from non-car images. The choice of SVC was made as it is a superior machine leearning algorithm when only two categories of training data are available. In this supervised learning environment, SVC provided the best of practicality (performance, speed) and efficiency (accuracy, prediction). 

Initially, the coombined feature vectors obtained using HOG, Spatial, and Histogram features are normalized using the `StandardScaler()` function from sklearn, so as to disallow the dominance of any one feature set over the other. Next, the car and non-car images are randomly split into a training and testing data set for the SVC to train and validate on. 

###Sliding Window Search

The code for this step can be found in the 12th cell of the notebook under the `find_cars()` function.

The basic principle behind the sliding windows technique is the use of a fixed size window that is made to slide a fixed distance (in units of cells) on each iteration. On each of these locations, the contents of the window are passed to the above-mentioned `extract_features()` function to aggregate the three feature vectors. This features vector is fed to the trained SVC and a prediction is made in the form of True (car identified) or False (car not identified). If a True Positive is found, the windows corners are appended to an array. Once the whole image is traversed, the final array with all True Positive windows is returned. 

A window overlap of 75% (2 cells in an 8 * 8 cell window) for the windows to traverse on each iteration in both the X and the Y directions. With a traversal rate of 16 pixels (two 8 * 8 cells), the speed was enough to not miss an object. A higher traversal rate (say 64 pixels) could potentially lead to the skipping of a far-away car in the image. A smaller rate was also tested, but resulted in a significantly slower process, and only a marginally better result. 

Here is an output for the aggregated windows -

![alt text][image3]

Another example - 

![alt text][image4]
---

### Video Pipeline

Here's a [link to my video result](./project_video_done.mp4)

The pipeline code can be found in the 14th cell of the notebook, under the `pipeline()` function. The function takes as input an image from the test video, and outputs the same image with bounding boxes drawn over identified objects. 

A few additional steps were taken to get elegant, stable and continuous bounding boxes -
1. Heatmaps - The use of heatmaps allowed me to remove any false positives where only a single windows was identified as True Positive (most car images would generally have multiple overlapping True Positive windows). In addition, heatmaps allowed me to label the cars, and hence, provide a contextual awareness as to the location of say Car1 with respect to Car2. Finally, heatmaps allowed me to identify 'blobs' of windows in an area and combine them into one bounding box instead of the several boxes seen above. 
2. Multiple window scales - For this project, I used the `find_cars()` function three times. I started with a small window size spanning a limited Y-range of ~ 400 to 475. This corresponds to the cars that appear smaller as they approach the road horizon. Next, a bigger scale of 1.2 is used on a larger Y-range as well. This corresponds to cars slightly ahead of the camera. And finally, a large scale of 1.5 is used as the window size that spans over the entire second half of the image height. This practice helped create a larger number of windows that were capable of identifying cars of all sizes appearing at all distances. It must be noted that although this approach is workable, a much better of doing this is by using progressively increasing scale values on a similarly increasing Y-range. 
3. Historical Frame data - The frames preceding the current one also hold some important information. In this project, I used heatmaps for the previous 5 frames in-order to better detect false positives by eliminating any windows that weren't identified in all 5 previous frames. In addition, I averaged the heatmap data for the last 5 frames to get smoother moving bounding boxes. 

The code for building heatmaps, thresholding, and drawing final bounding boxes can be found in cells 9, 10, and 11 respectively.

Here are five frames and their corresponding heatmaps:

![alt text][image5]

Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

The final output looks like this -
![alt text][image7]

---

###Discussion

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further. 

I believe a much more robust model can be created by using a better trained classifier, a progressively increasing windows size that corresponds with the nature of a car appearing bigger as we approach it, and finally, by using a multi-layer classifier that can distinguish between cars, bikes, pedestrains, traffic signs etc. 

The pipeline, although robust on a clear sunny day, may not work as spectacularly on a windy, snowy, or foggy day. The model hasn't been tested in night time images either. Also, I'd like to combine this project with the earlier "Behavioral Cloning" and "Advanced Lane Finding" Projects to build a Level 2 Autonomous vehicle. That would be pretty cool :)
