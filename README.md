# Fabricated Futures: GANs for Garment Design

> A deep learning project to generate fashion apparel images using GANs and the Fashion-MNIST dataset.

## 🧵 Project Overview

**FashionGAN** implements a Generative Adversarial Network (GAN) to synthesize realistic images of clothing items based on the Fashion-MNIST dataset. A convolutional neural network-based **generator** creates 28×28 grayscale images resembling fashion items, while a **discriminator** learns to differentiate real from generated images.

<p align="center">
  <img src="samples/epoch_50.png" alt="Sample output" width="300"/>
</p>

---

## 🗂 Dataset

- **Fashion-MNIST** from TensorFlow Datasets (`tfds`)
- 10 clothing categories (e.g., T-shirt, Dress, Sneaker, Bag)
- 28×28 grayscale images

**Preprocessing:**
- Normalize pixel values to [-1, 1]
- Shuffle, batch, and prefetch for performance

---

## 🧠 Model Architecture

### 🧵 Generator
- Input: Random noise vector (`latent_dim`)
- Layers: Dense → Reshape → Conv2DTranspose × 3
- Output: 28×28×1 grayscale image

### 🧵 Discriminator
- Input: Image (real or fake)
- Layers: Conv2D → LeakyReLU → Dropout × 2 → Flatten → Dense
- Output: Binary classification (real/fake)

---

## ⚙️ Training Setup

- **Training Data:**
  - Real: Fashion-MNIST
  - Fake: Generator outputs from random noise

- **Loss Functions:**
  - Generator: Binary crossentropy (goal: fool discriminator)
  - Discriminator: Binary crossentropy (real=1, fake=0)

- **Optimizers:** Adam for both Generator and Discriminator

- **Latent Vector:** Random normal vector of shape `(latent_dim,)`, e.g., 100

- **Label Trick:** Use `tf.ones_like()` and `tf.zeros_like()` for real/fake labels

- **Noise Injection (Optional):** Add label smoothing/noise for training stability

- **Training Loop:** Custom `train_step()` with `tf.GradientTape`

- **Model Subclassing:** `FashionGAN(tf.keras.Model)` for full control

- **Callbacks:** `GANMonitor` to save output samples after each epoch

---

## 📊 Metrics & Monitoring

- `d_loss`: Discriminator loss
- `g_loss`: Generator loss
- Sample images saved each epoch to visualize generator progress

---

## 🚀 Future Scope

- Train on higher-resolution datasets (e.g., DeepFashion)
- Integrate text-to-image generation for guided designs
- Use Conditional GANs for category-specific outputs
- Deploy as a tool for designers or fashion startups
- Experiment with attention mechanisms and StyleGANs


