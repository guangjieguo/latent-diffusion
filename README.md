# latent-diffusion
This is for an assignment for my Deep Learning class. What I did is doing latent diffusion from scratch. 
smallNORB Dataset is used to train the neural networks. The project has been done by following steps.

1. Use the neural network in example3autoenc.py and smallnorb.data to train a
model for copying images. Then, the saved net can be divided at the fully
connected layer of 128 neurons, the part after which is decoder, and the part
before which (inclusive) can be used as encoder. Therefore, the latent in this
case will be a 1D array with 128 elements. Please see the code in
autoencosmallnorb.py and encoder_smallnorb.py.

2. Generate noised images for training a denoising network. Firstly, extract a
type of images from smallnorb.data, and in my case all the plane images from
train_images and test_images are picked up. Secondly, generate images with
different number of noise layers. Thirdly, constitute dataset and labelset use
original images and noised images. Each image in dataset corresponds to a
label (also an image) in labelset, and the image is just the label but added a
layer of noise. Forthly, numpy save dataset and labelset to a file for further use. Please see the code in datagenerating_noised.py.

3. Put dataset and labelset into encoder_smallnorb.py to generate their latent.

4. Build a 1D CNN for denoising latent of noised images. Use the latant of
dataset as input and the latent of labelset as label. This is a regression task. Save the trained net for further use. Please see the code in denoising.py.

5. Now we can use the pre-saved autoenco net and denoising net to generate
plane image from a pure noise image. Firstly, create a (96,96) pure noise
image. Secondly, put it into the encoder to produce its latent. Thirdly, use
denoising net to denoise the latent. Forthly, an image can be produce by using
decoder. Check the image. If the image doesn’t show any signature of plane, go back to the third step to denoise the latent further and decode it. Continue
this loop until some kind of plane emerges, and save the image by hand if you
like. Please see the code in latent_diffusion.py and functions.py, and run
latent_diffusion.py to generate a plane image. BUT don’t expect too much. Only a blurry plane shape might emerge at most.

Please see generated_iamges



