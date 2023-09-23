import tensorflow_datasets as tfds
import numpy as np
import pickle, gzip
import os

"""
SmallNORB dataset consists of 96x96 black and white images of toys, two images from stereo camera taken under specific
lighting conditions, and camera position (specified by the azimuth and elevation).

There are 24300 images in the train and test set, equal number of samples per each class.

In numpy array format, the label of each image is 5-dimensional, with the dimensions corresponding to the following 
information:
- dim indexed at 0, integers in range 0-9 (inclusive) corresponding to the instance of given toy in the image 
  (so...there are 9 instances of each type of toy);
- dim indexed at 1, integers in range 0-17 (inclusive) corresponding to the azimuth of the cameras taking the photos (to
  get the actual azimuth value multiply the integer value by 20 for azimuth range from 0-340 degres);
  (0='four-legged animal',1='human',2='airplane',3='truck',4='car')
- dim indexed at 2, integers in range 0-4 (inclusive) - corresponding to the type/category of the toy 
  (0='four-legged animal',1='human',2='airplane',3='truck',4='car');
- dim indexed at 3, integers in range 0-8 (inclusive) - corresponding to the elevation of the cameras taking the photos
  (to get the actual elevation value, multiply the integer by 5 and add 30 degrees for elevation range from 30-70
  degrees);
- dim indexed at 4, integers in range 0-5 (inclusive) - corresponding to specific lighting condition.

To use this script in another, just drop this file in the same folder and then, you can invoke it from the other
script like so:

from load_smallnorb import load_smallnorb

(train_images, train_labels),(test_images,test_labels) = load_smallnorb()


"""


def load_smallnorb(format='numpy'):
   """
   Loads the smallNORB dataset
   Arguments:
       format: a string ('numpy' (default), 'tfds', or 'pandas')
   Returns:

      when format=='numpy':
         Two tuples (train_images, train_labels),(test_images,test_labels), where *_images is a 24300x96x96x2 numpy
         array of 243000 samples of two (that's the last 2 in the numpy shape) of 96x96 black and white images; the
         *_labels are a 24300x5 numpy array with *_labels[:,1] indicating the azimuths, *_labels[:,2] object types,
         *_labels[:,3] indicating the elevation, and *_labels[:,6] the lighting of the corresponding image
         (*_labels[:,0] specifies the instance of the toy in the image...which can be ignored);

      when format=='tfds':
         A tuple (smallnorb_train, smallnorb_test) containing the train and test dataset in tfds format;

      when format=='pandas':
         A tuple (smallnorb_train, smallnorb_test) containing the train and test dataset in pandas data frame format.
   """

   numpy_save = 'smallnorb.data'

   # If request format is 'numpy' and 'smallnorb.data' file exists,
   # load it (speeds up the loading).
   if format=='numpy' and os.path.isfile(numpy_save):
      with gzip.open(numpy_save) as f:
         (train_images, train_labels),(test_images,test_labels) = pickle.load(f)

      return (train_images, train_labels),(test_images,test_labels)

   # Load the smallnorb dataset in tfds format
   smallnorb  = tfds.load(name="smallnorb", split=None)
   smallnorb_train = smallnorb['train']
   smallnorb_test = smallnorb['test']

   # If tfds format requested, return the data
   if format == 'tfds':
      return smallnorb_train, smallnorb_test

   # Convert data to pandas data frame

   smallnorb_train = tfds.as_dataframe(smallnorb_train)
   smallnorb_test = tfds.as_dataframe(smallnorb_test)

   # If pandas format requested, return the data
   if format == 'pandas':
      return smallnorb_train, smallnorb_test

   # Conert pandas frame to numpy
   smallnorb_train = smallnorb_train.to_numpy()
   smallnorb_test = smallnorb_test.to_numpy()

   # Fetch the data out of the frames

   N = 24300
   train_images = np.zeros((N,96,96,2)).astype('uint8')
   train_labels = np.zeros((N,5)).astype('uint8')

   test_images = np.zeros((N,96,96,2)).astype('uint8')
   test_labels = np.zeros((N,5)).astype('uint8')


   for n in range(N):
      train_images[n,:,:,0] = smallnorb_train[n,0][:,:,0]
      train_images[n,:,:,1] = smallnorb_train[n,1][:,:,0]
      train_labels[n,:] = smallnorb_train[n,2:].astype('uint8')

   for n in range(N):
      test_images[n,:,:,0] = smallnorb_test[n,0][:,:,0]
      test_images[n,:,:,1] = smallnorb_test[n,1][:,:,0]
      test_labels[n,:] = smallnorb_test[n,2:].astype('uint8')

   # Save numpy data to 'smallnorb.data' for fast loading
   with gzip.open(numpy_save, 'w') as f:
      pickle.dump(((train_images, train_labels),(test_images,test_labels)), f)

   return (train_images, train_labels),(test_images,test_labels)

if __name__ == '__main__':
   import matplotlib.pyplot as plt

   #Load the data in the numpy format
   (train_images, train_labels),(test_images,test_labels) = load_smallnorb(format='numpy')

   category_labels = ['animal', 'human', 'airplane', 'truck', 'car']

   fig, axes = plt.subplots(3, 3, figsize=(8, 8))
   fig.subplots_adjust(hspace=0.2, wspace=0.1)


   for i, ax in enumerate(axes.flat):
      # Just show the left image
      image = train_images[i,:,:,0]
      # Fetch category
      category_str = category_labels[train_labels[i,2]]
      # Get the azimuth
      azimuth = train_labels[i,1]*20
      # Get the elevation
      elevation = train_labels[i,3]*5+30
      # Get the lighting
      lighting = train_labels[i,4]

      # Show image
      ax.imshow(image, cmap='gray')
      ax.text(0.5, -0.12, f'{category_str} (el={elevation},az={azimuth},lt={lighting})', ha='center',
           transform=ax.transAxes, color='black')
      ax.set_xticks([])
      ax.set_yticks([])

   plt.show()