# AABDW2
## Assignment 2: Identify swimming pools from satellite images

### 2.1 Problem Statement
For this assignment, the challenge was to construct a deep learning pipeline that
learns to predict the location of swimming pools found in satellite photos. For
this task, we received access to over 14.000 satellite images containing swimming
pools. Each image was guaranteed to contain at least one swimming pool, which
was located at the centre of the image. It was also possible that an image contained
multiple swimming pools, something which actually occurred frequently
in the given dataset.
In addition to the images, we also received metadata that describes the coordinate
points making up the bounding polygon of the central swimming pool
in the image. Note that in the case of multiple swimming pools appearing in one
image, the provided bounding polygon would only apply to the centrally located
swimming pool.
Of the two possible goals for this assignment, we opted for the more difficult
goal, i.e. to fully predict the bounding polygon of the swimming pool (as opposed
to simply predicting the location using an upright rectangle).

### 2.2 Methodology
General Requirements
In addition to fulfilling the chosen task of fully predicting the bounding polygon
of the swimming pool, we had to make sure that our approach fulfills certain
criteria:
– Prevent overfitting, such as correctly handling images where the swimming
pool is not located in the centre of the image.
– Correctly process images that do not contain any swimming pool at all.
– Allow input images of different sizes.
Approach Outline
We treated the task as a binary segmentation problem, which is an approach
that will assign labels to each individual pixel of the image. In this case, each
pixel would either be labelled as swimming pool or background.
One benefit of this approach is that the problem gets considerably simplified
in the sense that the model need not learn to grapple with irregular bounding
polygons, and can instead work on a per-pixel basis. Another consequence worth
mentioning is that a per-pixel approach can naturally deal with images containing
multiple pools, even if these pools would be located at the border of the
image, in which case they can sometimes partially fall outside of the image.
The case of multiple pools in one image merits some further discussion. In
fact, during our preliminary dataset analysis, we noticed that a significant number
of images contained two or more pools. Normally this would not pose a
challenge to the binary segmentation approach that we employed, were it not
for the fact that each image only had one bounding polygon that was provided
as ground truth (delineating the swimming pool at the centre of the image),
irrespective of the actual number of swimming pools contained in the image.
As a result, when training the model, it will be actively punished for correctly
identifying pixels belonging to swimming pools that are not at the centre of the
image (and thus do not correspond to the provided ground truth).We will briefly
discuss how we attempted to mitigate the problem of inaccurate ground truth
when we talk about our dataset preprocessing method.

#### Model Used
For the deep learning model, we made use of the Segmentation Models library,
which is available on GitHub [12]. We chose the U-Net model architecture [9],
shown in figure 4, which is widely used for image segmentation. We combined
this with the efficientnetb3 [11] backbone, which is pretrained on the ImageNet
dataset [5] and acts as the feature extractor for this Deep Learning pipeline.
As can be seen in Figure 4, the model follows an encoder-decoder structure,
meaning that it forces the input through a lower-dimensional bottle-neck,
and outputs an image of size equivalent to the input. This output contains the
segmentation map of the class that the model is trained to predict.
These days it is considered standard to use convolutional layers in any deep
learning architecture that processes images, and so this model is no exception.
Note that one advantage of using convolutional layers lies in the fact that the use
of a convolutional kernel (which is usually much smaller than the input image)
prevents the model from easily associating certain regions of an image with a
certain class label. In our case, this helps to guarantee that the model will not
overfit to swimming pools consistently appearing at the centre of the image.
Data augmentation and preprocessing
Another standard practice when applying Deep Learning to an image-based
task, is to use data augmentation to allow the model to generalise better and to
prevent it from overfitting. For this part, we used the Albumentations [2] library,
which provides all the types of data augmentation that we deemed necessary for
this task. In the end we applied a large variety of augmentations, so we will only
list some of them here:
– Horizontal or vertical flip
– Upscaling or downscaling, the latter possibly combined with shifting
– Rotation
– Cropping (combined with resizing)
– Adding gaussian noise
– ...
Note that we add padding to the augmented image where needed. An exemplary
augmented image can be seen figure 5.
While these are all standard data augmentation techniques, we do want to
highlight the fact that image cropping-resizing and shifting is especially relevant
to this use case. The image cropping and resizing is useful in images containing
multiple swimming pools. As we mentioned earlier, the provided ground truth
only applies to the centrally-located swimming pool, which can be a source
of confusion to the model during training. Cropping the image has a chance
of cutting out other swimming pools in the image, thus removing this source
of confusion. Image shifting, on the other hand, once again makes sure that
swimming pools are not necessarily located at the center of the image, further
reducing the chances of a model overfitting on this data artefact.
Next to the data augmentation, we also had to apply some other type of preprocessing
in order to adapt the training data to our chosen segmentation task.
For this, we extracted the information on the bounding polygon for each image
from the provided dataset metadata, and labelled each pixel in the bounding
polygon as being part of a swimming pool. Note that our chosen deep learning
model deals with variable-sized inputs out of the box as part of its internal
preprocessing methodology, so we did not need to implement this ourselves.

### 2.3 Experimental Results and Evaluation
We split the dataset in a training-validation-test split of 70-15-15. During training,
we used the validation set to perform hyper-parameter optimization.
For our loss function we used a combination of dice loss and focal loss, the
latter of which is a version of cross entropy that is well-suited to deal with
class imbalance. The class imbalance in this case is caused by the low ratio of
swimming pool to background pixels. We also used the Adam optimizer with
a learning rate of 0.0001. The activation function was sigmoid. We ended up
having to use a batch size of 6 due to the GPU RAM constraints of Google
Colab. We trained the model for a total of 5 epochs.
As an evaluation metric we used the IoU (Intersection over Union) score, a
popular choice for image segmentation tasks, as it punishes all types of wrong
predictions, such as predicting the wrong area, predicting too large or too small
an area, etc.
After the model configured using the above hyper-parameters, it attained an
IoU score of 0.5172 on the test set. At first glance, this could hardly be considered
a good score. However, this metric doesn’t paint the full picture of the problem
at hand. We will explain this further with some examples.
For figure 6 we have selected several successful and less successful model predictions.
One thing that is immediately clear is that the ground truth provided
in the Target image is often too simplistic or not fully accurate. In figure 6a
and figure 6d we see that, compared to the target image, our model’s prediction
is visually closer to the shape of the swimming pool of the original image. We
also see in figure 6e that our model has no trouble predicting locations of black
swimming pools, and that it can also handle multiple swimming pools in one
image.
However, figures 6b and 6c show that the model seemingly still struggles
with pools located at the edge of the image (which are sometimes only partially
visible). Figure 6c also shows an example of the model predicting a (partial)
pool in a location where there is none.

#### Inaccurate ground truth
At this point, based on figure 6 we already have an indication that the low IoU
score is caused by the model predicting second swimming pools that are not part
of the ground truth. A way to test this hypothesis is by looking at the recall score
obtained on the test set. The recall score will give a good indication of whether
the model was able to accurately detect the swimming pool at the centre of the
image, regardless of false positives.
We found that the model attained a recall score of 0.828 on the test set. While
this score is significantly better than the IoU score and lends some credibility
to our initial hypothesis, it is still not very high. As part of further manual
investigation of the training data, we found that the provided bounding polygon
often doesn’t accurately delineate the swimming pool located at the centre, an
example of which is shown in figure 7. In this figure, we see that the model’s
prediction is actually more accurate than the provided target. The number of
false negative pixels in these examples is quite large, which drags down the recall
score significantly. This trend can also be seen when looking in detail at all the
predictions on the test set. In figure 8 the distribution of the results can be seen.
The 300 images with a recall of under 0.4 were manually inspected, and it was
found that almost all of them have a significant misaligned ground truth.

### 2.4 Discussion
While our model did manage to seemingly perform a better task than the provided
ground truth, one important take-away from this work is that it is very
difficult to produce a well-performing model if the underlying training data contains
a significant number of mislabelled instances. For this task, our model
specifically struggled with misaligned and inaccurate bounding polygons of the
centrally located swimming pool, as well as images with multiple pools where
anything but the central swimming pool was not part of the provided ground
truth. This last fact may help explain why, despite using convolutional neural
networks and applying a myriad of data augmentation techniques, the model
was still apparently taught to ignore swimming pools that are located at the
edge of an image.
Fixing the ground truth of the training data seems like the most logical
next step, though this would fall outside of the scope of this work. Given that
manual verification of our model’s predictions suggests that our model actually
outperforms the ground truth, one possibility would be to discard the provided
bounding polygon and instead use our model’s predictions as the target for a
new training run. This kind of approach could be applied iteratively until manual
analysis reveals that no further improvement is required.
