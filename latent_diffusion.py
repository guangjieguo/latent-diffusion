import numpy as np
import matplotlib.pyplot as plt
import functions

'''create a pure noise-image'''
zeros_image = np.zeros((1,96,96,1)).astype('uint8')
noise_image = np.zeros((1,96,96,1)).astype('uint8')
noise_image[0,:,:,0] = functions.addnoise(zeros_image[0,:,:,0], 9216)
plt.imshow(noise_image[0,:,:,0], cmap='gray')
print("Close the plot to continue...")
plt.show()

'''encode the noise-image'''
latent_noise_image = functions.encoding(noise_image)

'''denoise the latent, decode the denoised latent, and plot the image'''
denoised_latent = latent_noise_image
image = functions.decoding(denoised_latent)
plt.imshow(image[0,:,:,0], cmap='gray')
print("Close the plot to continue...")
plt.show()
iterate_num = 70
for i in range(iterate_num):
    denoised_latent = functions.denoising(denoised_latent)
    image = functions.decoding(denoised_latent)
    plt.imshow(image[0, :, :, 0], cmap='gray')
    print("Close the plot to denoise further...")
    plt.show()


