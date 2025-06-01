# Intro to AI: Denoising Autoencoder

End-to-end pipeline:

1. Generate synthetic noisy/text-overlaid images
2. Train a denoising autoencoder
3. Apply the autoencoder to denoise images
4. Detect outlines via OpenCV, refine line coordinates with LSTM, output to .docx
5. Evaluate restoration quality using SSIM

Key components:

- `gen_data.py`: create noisy/text images
- `dataset.py`, `data.py`: data loading & preprocessing
- `AutoEncoder.py`, `UNet.py`, `line_refinement_model.py`: model definitions
- `train.py`, `main.py`: training & inference
- `recover_outline.py`: apply model to new images
- `ssim_eval.py`: compute SSIM (original vs. noisy/denoised)

# Usage:

1. Clone the repository and enter the project folder

   ```bash
   git clone https://github.com/poyu1234/intro2ai-final-project-group1
   cd intro2ai-final-project-group1
   ```

2. (Optional) Create and activate a virtual environment

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies

   ```bash
   pip install -r requirements.txt
   ```

4. Generate synthetic noisy/text-overlaid images

   ```bash
   python gen_data.py
   ```

5. Train the denoising autoencoder

   ```bash
   python train.py
   ```

6. Train the LSTM line refinement model

   ```bash
   python line_refinement_model.py
   ```

   Note: This step is optional if you only want to denoise images without refining outlines.

7. Denoise new images with the trained autoencoder

   ```bash
   python main.py
   ```

8. Evaluate restoration quality using SSIM
   ```bash
   python ssim_eval.py
   ```

# Environment:

- OS: macOS / Linux / Windows
- Python: ≥ 3.8
- GPU support (optional): CUDA 11.x + cuDNN
- Packages: listed in `requirements.txt`

# Hyperparameters:

Autoencoder (`AutoEncoder.py`):

- learning_rate: 1e-3
- batch_size: 64
- epochs: 100
- latent_dim: 256
- input_size: [128, 128, 1]
- noise_level: 0.2

LSTM line refinement (`model.py`):

- R (max coordinate): 100
- epochs per cycle: 50
- batch_size: 500
- line count N: 10 (range [5–15])
- penalty_factor: 100.0

data_prep (`gen_data.py`):

- num_samples = 100
- N = 4
- image_size = 256
- add_text = True
- line_width_range = (3,5)

main (`main.py`):

- input_folder = {your_noisy_img_folder}
- output_docx_folder = {your_output_docx_folder}
- visualization_folder = {your_output_visualization_folder}
- denoised_folder = {your_output_denoised_img_folder}
- model_weights_path = {your_LSTM_model_path}
- ae_model_path = {your_autoencoder_model_path}
- un_model_path = {your_UNet_model_path}
- denoise_model = 'AutoEncoder' or 'UNet'

# References:

- Vincent, P. et al. “Stacked Denoising Autoencoders,” Journal of Machine Learning Research, 2008.
- Hochreiter, S. & Schmidhuber, J. “Long Short-Term Memory,” Neural Computation, 1997.
- Wang, Z. et al. “Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM),” IEEE TIP, 2004.
- TensorFlow Keras Guide: https://www.tensorflow.org/guide/keras
- OpenCV Python Tutorials: https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
- python-docx Documentation: https://python-docx.readthedocs.io/
