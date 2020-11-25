# This is my solution to problem 7

## What is found in this directory:
First, there is a dataset folder where the dataset is located.

Second, there is an img folder where all the plots are located. Further, in the img folder there are separate folders for pca, mds and isomap plots.

The pca folder contains two images. Both are the exact same visualization of the manifold, but one is with annotations and one is without annotations.

The mds folder contains four images. There are essentially two different manifold plots. One with feature importance, and one without. For each manifold plot, there is also a version with annotations included.

The isomap folder contains 95 different manifold plots, corresponding to how many neighbors are allowed. For each version there is also an annotated version.

With annotations I mean that each point also has its name written next to it.

Finally, there is the main python script.

The script requires the modules numpy, matplotlib, sklearn and argparse to run.

## How to generate the different images:
Besides looking at the images in the img file, the images can also be generated from the python script.

To view the plot with PCA as dimensionality reduction method, simply run:
python problem_7.py --PCA

To view the plot with MDS as dimensionality reduction method, simply run:
python problem_7.py --MDS

To view the plot with MDS with feature importance, simply run:
python problem_7.py --MDS --importance

To view the plot with isomap as dimensionality reduction method, simply run:
python problem_7.py --isomap (note that this will attempt to plot 95 images one at a time, while the other will only plot 1 image)

Toview any of the plots above with annotations, include the --annotations flag.

This assumes also that "python" command refers to python3.
