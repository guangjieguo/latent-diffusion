import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import pickle, gzip
from load_smallnorb import load_smallnorb


# Download data
(train_images, train_labels),(test_images,test_labels) = load_smallnorb()

load_from_file = True

# Create 'saved' folder if it doesn't exist
if not os.path.isdir("saved"):
   os.mkdir('saved')

# Specify the names of the save files
save_name = os.path.join('saved', 'smallborbautoenc')
net_save_name = save_name + '_cnn_net.h5'
history_save_name = save_name + '_cnn_net.hist'

# Get the dimensions of the input - first
# dimension is number of point, so it's ignored,
# H is the height in pixel, W is the weight in pixels # C is the number of colour channels
_,H,W,C = np.shape(train_images[:, :, :, 0:1])

if load_from_file and os.path.isfile(net_save_name):
   # ***************************************************
   # * Loading previously trained neural network model *
   # ***************************************************

   # Load the model from file
   print("Loading neural network from %s..." % net_save_name)
   net = tf.keras.models.load_model(net_save_name)

   # Load the training history - since it should have been created right after
   # saving the model
   if os.path.isfile(history_save_name):
      with gzip.open(history_save_name) as f:
         history = pickle.load(f)
   else:
      history = []
else:
   # ************************************************
   # * Creating and training a neural network model *
   # ************************************************

   # Create feed-forward network
   net = tf.keras.models.Sequential()

   # Add a convolutional layer, 3x3 window, 64 filters - specify the size of the input as 96x96x1, padding="same"
   net.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1,1), activation='relu',
                                  input_shape=(H, W, C),padding='same'))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2),strides=(2,2)))

   # Add a convolutional layer, 3x3 window, 128 filters, padding="same"
   net.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1,1), activation='relu',padding="same"))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))

   # Add a convolutional layer, 3x3 window, 256 filters
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1,1), activation='relu'))

   # Add a max pooling layer, 2x2 window
   # (implicit arguments - padding="valid")
   net.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(1,1)))

   # Flatten the output maps for fully connected layer
   net.add(tf.keras.layers.Flatten())

   # Add a fully connected layer of 128 neurons
   net.add(tf.keras.layers.Dense(units=128, activation='relu'))

   # Add a fully connected layer of 512 neurons
   net.add(tf.keras.layers.Dense(units=512, activation='relu'))

   # Add a fully connected layer with number of output neurons the same
   # as the total number of attributs of the input height*width of the
   # image times the number of channels, note the activation is
   # linear, not softmax
   net.add(tf.keras.layers.Dense(units=H * W * C, activation='relu'))

   # Reshape the H*W*C output vector to a HxWxC tensor, to match
   # the NxHxWxC shape of the x_train dataset (remember, N is the number
   # of samples, so it's ignored for the purpose fo reshaping the final layer)
   net.add(tf.keras.layers.Reshape((H, W, C)))

   # Define training regime: type of optimiser, loss function to optimise and type of error measure to report during
   # training
   net.compile(optimizer='adam',
               loss='mse')

   # Train the model for 50 epochs, using 33% of the data for validation measures,
   # shuffle the data into different batches after every epoch
   train_info = net.fit(train_images[:, :, :, 0:1], train_images[:, :, :, 0:1],
                        validation_split=0.33, epochs=50, shuffle=True)

   # Save the model to file
   print("Saving neural network to %s..." % net_save_name)
   net.save(net_save_name)

   # Save training history to file
   history = train_info.history
   with gzip.open(history_save_name, 'w') as f:
      pickle.dump(history, f)

   # *********************************************************
   # * Training history *
   # *********************************************************

   # Plot training and validation accuracy over the course of training
   if history != []:
      fh = plt.figure()
      ph = fh.add_subplot(111)
      ph.plot(history['loss'], label='mse')
      ph.plot(history['val_loss'], label='val_mse')
      ph.set_xlabel('Epoch')
      ph.set_ylabel('MSE loss')
      ph.legend(loc='lower right')

# Compute output for 9 test images
y_test = net.predict(test_images[0:9, :, :, 0])

# The output will be a 9xHxWxC tensor of values
# somewhere between 0 and 255 (because the desired
# output was just input, which are pixel values between
# 0 and 255...so here we convert y_test to an tensor
# of uint8's ...which are positive integer values between
# 0 and 255; anything in between is rounded up or down,
# anything outside of that range will be rounded to 0 or 255.
y_test = y_test.astype('uint8')

# Show decoded images from 9 test images
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.subplots_adjust(hspace=0.2, wspace=0.1)
for i, ax in enumerate(axes.flat):
    image = y_test[i, :, :]
    ax.imshow(image, cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
plt.show()

