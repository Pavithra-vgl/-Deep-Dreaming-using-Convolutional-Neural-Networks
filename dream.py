import sys, os
import numpy as np
import PIL.Image
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model

#thread limits
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

#Calculation of loss
class SquaredSumLoss(tf.keras.losses.Loss):
    def call(self, y_true, y_pred):
        return tf.reduce_sum(tf.square(y_pred))

if __name__ == "__main__":
    #argument check
    if len(sys.argv) != 4:
        print("Usage: python3 dream.py <image> <X> <N>")
        sys.exit(1)
    #Parsing arguments
    image_path, X, N = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

    #Loading and normalising the images
    img = PIL.Image.open(image_path).convert("RGB").resize((224, 224))
    img = np.array(img, dtype=np.float32) / 255.0

    #dynamic height and width
    input_tensor = Input(shape=(None, None, 3))

    #Loading pretrained mobilenetv2
    model = tf.keras.applications.MobileNetV2(
        input_tensor=input_tensor, include_top=False, weights="imagenet"
    )

    #Printing layers with names
    for i, l in enumerate(model.layers):
        print(i, l.name)

    #Validation 
    if not (0 <= X < len(model.layers)):
        raise ValueError(f"X must be between 0 and {len(model.layers)-1}")

    #Creation of model
    dream_model = Model(input_tensor, model.layers[X].output)
    loss_func = SquaredSumLoss()

    #output filenames
    base = os.path.basename(image_path)
    name, ext = os.path.splitext(base)
    final_out = f"dream_{name}{ext or '.png'}"

    #Dream step size
    step = 0.008

    octaves = 3
    octave_scale = 1.2

    #creates trainable image tensor
    image = tf.Variable(img[None, ...], dtype=tf.float32)

    
    for o in range(octaves):
        #Gradient ascent loops
        for i in range(N):
            with tf.GradientTape() as g:
                output = dream_model(
                    tf.keras.applications.mobilenet_v2.preprocess_input(image * 255.0)
                )
                loss = loss_func(0.0, output)

            #derivative of loss w.r.t input pixels
            grad = g.gradient(loss, image)
            #Normalising gradient 
            image.assign_add(step * grad / (tf.math.reduce_std(grad) + 1e-8))
            image.assign(tf.clip_by_value(image, 0.0, 1.0))

            if (i + 1) % 10 == 0:
                plt.imsave(f"dream_{name}_oct{o+1}_step{i+1:04d}.png", image.numpy()[0])

        if o < octaves - 1:
            new_h = int(image.shape[1] * octave_scale)
            new_w = int(image.shape[2] * octave_scale)
            image = tf.Variable(tf.image.resize(image, (new_h, new_w)), dtype=tf.float32)

    final_img = tf.image.resize(image, (224, 224)).numpy()[0]
    plt.imsave(final_out, final_img)
    print("Saved:", final_out)
