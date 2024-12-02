import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.models import Model
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Helper Functions
def load_and_process_image(image_path, target_size=(224, 224)):
    img = Image.open()
    img = img.resize(target_size)
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img

def deprocess_image(processed_img):
    img = processed_img.copy()
    img = img.reshape((img.shape[1], img.shape[2], 3))
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    img = img[:, :, ::-1]
    img = np.clip(img, 0, 255).astype('uint8')
    return img

# Compute Content Loss
def compute_content_loss(base_content, target):
    return tf.reduce_mean(tf.square(base_content - target))

# Compute Style Loss
def gram_matrix(input_tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)
    num_locations = tf.cast(input_shape[1] * input_shape[2], tf.float32)
    return result / num_locations

def compute_style_loss(base_style, gram_target):
    gram_style = gram_matrix(base_style)
    return tf.reduce_mean(tf.square(gram_style - gram_target))

# Load the VGG19 Model
def get_model():
    vgg = vgg19.VGG19(weights="imagenet", include_top=False)
    vgg.trainable = False
    content_layers = ["block5_conv2"]
    style_layers = [
        "block1_conv1",
        "block2_conv1",
        "block3_conv1",
        "block4_conv1",
        "block5_conv1",
    ]
    content_outputs = [vgg.get_layer(name).output for name in content_layers]
    style_outputs = [vgg.get_layer(name).output for name in style_layers]
    model_outputs = style_outputs + content_outputs
    return Model(vgg.input, model_outputs)

# Style Transfer Class
class StyleTransfer:
    def __init__(self, content_path, style_path, content_weight=1e4, style_weight=1e-2):
        self.content_image = load_and_process_image(content_path)
        self.style_image = load_and_process_image(style_path)
        self.generated_image = tf.Variable(self.content_image, dtype=tf.float32)
        self.model = get_model()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.optimizer = tf.optimizers.Adam(learning_rate=5.0)
        self.style_layers = 5
        self.content_layers = 1

    def compute_loss(self):
        model_outputs = self.model(tf.concat([self.style_image, self.content_image, self.generated_image], axis=0))
        style_features = model_outputs[:self.style_layers]
        content_features = model_outputs[self.style_layers:]
        
        # Style loss
        style_loss = tf.add_n([compute_style_loss(style_features[i], gram_matrix(style_features[i]))
                               for i in range(self.style_layers)])
        style_loss *= self.style_weight / self.style_layers
        
        # Content loss
        content_loss = compute_content_loss(content_features[-1], content_features[-1])
        content_loss *= self.content_weight / self.content_layers
        
        # Total loss
        total_loss = style_loss + content_loss
        return total_loss

    def train_step(self):
        with tf.GradientTape() as tape:
            loss = self.compute_loss()
        grad = tape.gradient(loss, self.generated_image)
        self.optimizer.apply_gradients([(grad, self.generated_image)])
        self.generated_image.assign(tf.clip_by_value(self.generated_image, -1.0, 1.0))

    def transfer_style(self, epochs=1000):
        for epoch in range(epochs):
            self.train_step()
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {self.compute_loss().numpy()}")
        return deprocess_image(self.generated_image.numpy())

# Main Code to Execute Style Transfer
if __name__ == "__main__":
    content_image_path = "content.jpg"  # Path to your content image
    style_image_path = "style.jpg"    # Path to your style image

    style_transfer = StyleTransfer(content_image_path, style_image_path)
    result = style_transfer.transfer_style(epochs=1000)

    plt.imshow(result)
    plt.axis('off')
    plt.show()
