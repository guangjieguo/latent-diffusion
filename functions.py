import tensorflow as tf
import os
import random


# define a function adding noise
def addnoise(image, num):
    result = image
    # 生成从0到9215的整数列表
    int_list = list(range(9216))
    # 从整数列表中随机选择num个不重复的数
    random_ints = random.sample(int_list, num)
    for i in random_ints:
        # 求10除以3的商和余数
        quotient, remainder = divmod(i, 96)
        random_int = random.randrange(256)
        result[quotient, remainder] = random_int
    return result

# define a function for encoding
def encoding(image):
    # load trained smallnorbautoenc net file name
    load_name = os.path.join('saved', 'smallborbautoenc')
    net_load_name = load_name + '_cnn_net.h5'
    enc_net = tf.keras.models.load_model(net_load_name)
    enc_net.pop()
    enc_net.pop()
    enc_net.pop()
    latent_image = enc_net.predict(image[:,:,:,0])
    return latent_image

# define a function for denoising
def denoising(latent):
    load_name = os.path.join('saved', 'denoising')
    net_load_name = load_name + '_cnn_net.h5'
    denoi_net = tf.keras.models.load_model(net_load_name)
    denoised_latent = denoi_net.predict(latent)
    return denoised_latent

# define a function for encoding
def decoding(latent):
    load_name = os.path.join('saved', 'smallborbautoenc')
    net_load_name = load_name + '_cnn_net.h5'
    net = tf.keras.models.load_model(net_load_name)
    dec_net = tf.keras.models.Sequential()
    dec_net.add(tf.keras.layers.Flatten(input_shape=(128,)))
    i = 0
    for l, layer in enumerate(net.layers):
        i = i + 1
        if i > 8:
            dec_net.add(layer)
    image = dec_net.predict(latent)
    return image
