import tensorflow as tf
import os
import pickle, gzip


load_from_file = True

# Specify the names of the save files
save_name = os.path.join('saved', 'denoising')
net_save_name = save_name + '_cnn_net.h5'

if load_from_file and os.path.isfile(net_save_name):
    print("The denoising network has been pre-saved.")

else:
    # ************************************************
    # * Creating and training a neural network model *
    # ************************************************

    # loading train_data
    raw_data_name = os.path.join('saved', 'latent_planes.pkl')
    print("Loading data from %s..." % raw_data_name)
    with gzip.open(raw_data_name, 'rb') as f:
        input_data1, label_data = pickle.load(f)

    # Build a Conv1D network
    # Input layer
    input_data = tf.keras.layers.Input(shape=(128, 1))

    # Add Conv1D layers
    conv_output1 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu',
                                          padding='same')(input_data)
    conv_output11 = tf.keras.layers.Conv1D(filters=64, kernel_size=3, strides=1, activation='relu',
                                           padding='same')(input_data)
    # Add Pooling layer
    pool_output1 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv_output11)

    # Add Conv1D layers
    conv_output2 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu',
                                          padding='same')(pool_output1)
    conv_output22 = tf.keras.layers.Conv1D(filters=128, kernel_size=3, strides=1, activation='relu',
                                           padding='same')(pool_output1)
    # Add Pooling layer
    pool_output2 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv_output22)

    # Add Conv1D layers
    conv_output3 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='relu',
                                          padding='same')(pool_output2)
    conv_output33 = tf.keras.layers.Conv1D(filters=256, kernel_size=3, strides=1, activation='relu',
                                           padding='same')(pool_output2)
    # Add Pooling layer
    pool_output3 = tf.keras.layers.MaxPooling1D(pool_size=2)(conv_output33)

    # Add a flatten layer
    flatten_output = tf.keras.layers.Flatten()(pool_output3)

    # Add a fully connected layer of 128 neurons
    dense_output1 = tf.keras.layers.Dense(units=128, activation='relu')(flatten_output)

    # Add a fully connected layer of 512 neurons
    dense_output2 = tf.keras.layers.Dense(units=512, activation='relu')(dense_output1)

    # Output layer
    output = tf.keras.layers.Dense(units=128, activation='relu')(dense_output2)

    # Collect together
    net = tf.keras.models.Model(inputs=input_data, outputs=output)

    # Show the structure
    net.summary()
    # Define training regime: type of optimiser, loss function to optimise during training
    net.compile(optimizer='adam',
                loss='mse')

    # Train the model for 30 epochs, using 33% of the data for validation measures,
    # shuffle the data into different batches after every epoch
    train_info = net.fit(input_data1, label_data, validation_split=0.33, epochs=30, verbose=1, shuffle=True)

    # Save the model to file
    print("Saving neural network to %s..." % net_save_name)
    net.save(net_save_name)

    print("The denoising network has been saved.")


