import tensorflow as tf
import os
import pickle, gzip

# Trained smallnorbautoenc net file name
save_name = os.path.join('saved', 'smallborbautoenc')
net_save_name = save_name + '_cnn_net.h5'

# Images file name
numpy_save = "dataset_noised_planes1500.data"

# Load the model from file
print("Loading neural network from %s..." % net_save_name)

# Load pre-saved net and take the encoding part
enc_net = tf.keras.models.load_model(net_save_name)
enc_net.pop()
enc_net.pop()
enc_net.pop()

# Show net's structure
enc_net.summary()

# load images from file
print("Loading data from %s..." % numpy_save)
with gzip.open(numpy_save, 'rb') as f:
    input_data, label_data = pickle.load(f)

# Creating letant
input_data_latent = enc_net.predict(input_data[:, :, :, 0])
label_data_latent = enc_net.predict(label_data[:, :, :, 0])

# Save latent to a file
save_name = os.path.join('saved', 'latent_planes.pkl')
with gzip.open(save_name, 'wb') as tf:
    pickle.dump((input_data_latent, label_data_latent), tf)

