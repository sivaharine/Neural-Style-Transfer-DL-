🧠 Neural Style Transfer using VGG19

This project performs Neural Style Transfer (NST) — combining the content of one image with the artistic style of another using deep learning.

It utilizes the VGG19 pre-trained model from TensorFlow to extract feature maps from multiple layers, capturing both content and texture details.

⚙️ Key Features

Image Preprocessing & Deprocessing: Converts input images into VGG-compatible format and reconstructs the stylized output back to RGB.

Loss Functions:

Content Loss — preserves structure from the content image.

Style Loss — matches textures and colors using Gram matrices.

Optimization: Uses the Adam optimizer with gradient descent to minimize total loss and iteratively improve the generated image.

Customizable Parameters: Control the balance between content and style using content_weight and style_weight.

🚀 Highlights

Implements an object-oriented approach through the StyleTransfer class for better modularity.

Combines computer vision, transfer learning, and image generation concepts.

Produces high-quality stylized images that blend structure and artistic patterns effectively.
