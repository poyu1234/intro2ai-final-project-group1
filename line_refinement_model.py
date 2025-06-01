import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM # Corrected imports
import numpy as np

R = 100 # Max coordinate for generated lines, also used in my_loss
epochs = 50

def generate_horizontal_vertical_lines(num_samples=1000, N=10):
    X = []
    Y = []

    for _ in range(num_samples):
        rectangles = [(0, 0, R, R)] 
        lines = []

        while len(rectangles) < N: # N rectangles means N-1 splits
            if not rectangles: break # Should not happen if N > 0
            idx = random.randint(0, len(rectangles) - 1)
            x1, y1, x2, y2 = rectangles.pop(idx)

            can_split_vertically = (x2 - x1) > 1
            can_split_horizontally = (y2 - y1) > 1

            if can_split_vertically and (not can_split_horizontally or random.choice(["vertical", "horizontal"]) == "vertical"):
                split = random.randint(x1 + 1, x2 - 1)
                rectangles.append((x1, y1, split, y2))
                rectangles.append((split, y1, x2, y2))
                lines.append((split, y1, split, y2)) # Vertical line
            elif can_split_horizontally:
                split = random.randint(y1 + 1, y2 - 1)
                rectangles.append((x1, y1, x2, split))
                rectangles.append((x1, split, x2, y2))
                lines.append((x1, split, x2, split)) # Horizontal line
            else:
                rectangles.append((x1, y1, x2, y2))
                if len(rectangles) + len(lines) >= N + N -1 : # Heuristic to avoid infinite loop if N is too large for small R
                    break


        # Add boundary lines
        lines.append((0, 0, R, 0))  # Top
        lines.append((0, 0, 0, R))  # Left
        lines.append((R, 0, R, R))  # Right
        lines.append((0, R, R, R))  # Bottom

        original_sample = lines
        noisy_sample = []

        for line_coords in original_sample:
            x1_orig, y1_orig, x2_orig, y2_orig = line_coords

            # Noise is +/- 1 pixel or 0
            noise_x1 = random.choice([-1, 0, 1]) if x1_orig != x2_orig else 0
            noise_y1 = random.choice([-1, 0, 1]) if y1_orig != y2_orig else 0
            noise_x2 = random.choice([-1, 0, 1]) if x1_orig != x2_orig else 0
            noise_y2 = random.choice([-1, 0, 1]) if y1_orig != y2_orig else 0
            
            x1_noisy = x1_orig + noise_x1
            y1_noisy = y1_orig + noise_y1
            x2_noisy = x2_orig + noise_x2
            y2_noisy = y2_orig + noise_y2

            noisy_sample.append((x1_noisy, y1_noisy, x2_noisy, y2_noisy))

        X.append(noisy_sample)
        Y.append(original_sample)

    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def my_loss(y_true, y_pred):
    mse_loss = tf.reduce_mean(tf.square(y_true - y_pred))
    
    x1, y1, x2, y2 = tf.split(y_pred, 4, axis=-1) # y_pred is normalized (0-1)

    normalized_threshold = 1.0 / R / 2.0

    is_horizontal = tf.abs(y1 - y2) < normalized_threshold
    is_vertical = tf.abs(x1 - x2) < normalized_threshold
    is_valid_line = tf.logical_or(is_horizontal, is_vertical)

    # Penalty: large value if not a valid line, zero otherwise
    penalty_factor = 100.0 # Penalty strength
    penalty = tf.where(is_valid_line, tf.zeros_like(mse_loss), tf.ones_like(mse_loss) * penalty_factor)
    penalty_loss = tf.reduce_mean(penalty)

    return mse_loss + penalty_loss

def draw_ascii(lines_to_draw, grid_R_scale=R):
    display_size = int(grid_R_scale + 1) # Accommodate coordinate R
    grid = [[" ." for _ in range(display_size)] for _ in range(display_size)]

    for line_coords in lines_to_draw:
        x1, y1, x2, y2 = map(int, line_coords) # Ensure integer coordinates for indexing

        # Clip coordinates to be within grid boundaries
        x1_c = max(0, min(x1, display_size - 1))
        y1_c = max(0, min(y1, display_size - 1))
        x2_c = max(0, min(x2, display_size - 1))
        y2_c = max(0, min(y2, display_size - 1))

        if x1_c == x2_c:  # Vertical line
            for y_coord in range(min(y1_c, y2_c), max(y1_c, y2_c) + 1):
                if 0 <= y_coord < display_size and 0 <= x1_c < display_size:
                    grid[y_coord][x1_c] = " #"
        elif y1_c == y2_c:  # Horizontal line
            for x_coord in range(min(x1_c, x2_c), max(x1_c, x2_c) + 1):
                if 0 <= y1_c < display_size and 0 <= x_coord < display_size:
                    grid[y1_c][x_coord] = "##"
    
    # Print grid (0,0 is top-left)
    print(f"Displaying on a {display_size}x{display_size} grid (approx):")
    for row in grid:
        print("".join(row))

# Define the Keras Sequential model
model = tf.keras.Sequential([
    tf.keras.layers.LSTM(128, return_sequences=True, input_shape=(None, 4)), # (timesteps, features)
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.Dense(4) # Output layer with 4 units for (x1, y1, x2, y2)
])

# Compile the model with the custom loss function
model.compile(optimizer='adam', loss=my_loss)

# Define weights file path
weights_file = 'm.weights.h5'

# Load weights if the file exists (e.g., to resume training or for inference)
if os.path.exists(weights_file):
    print(f"Loading weights from {weights_file}")
    try:
        model.load_weights(weights_file)
    except Exception as e:
        print(f"Could not load weights: {e}. Starting with an uninitialized model.")

if __name__ == '__main__':
    print(f"TensorFlow version: {tf.__version__}")
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        print(f"Found GPU(s): {physical_devices}")
        # Optional: Set memory growth to avoid allocating all GPU memory at once
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Could not set memory growth: {e}")
    else:
        print("No GPU found, training will use CPU.")

    try:
        while True: # Continuous training loop
            print("\\nGenerating training and testing data...")
            X_train, y_train = generate_horizontal_vertical_lines(num_samples=5000, N=random.randint(5, 15)) # Reduced samples for faster cycle
            X_test, y_test = generate_horizontal_vertical_lines(num_samples=1000, N=random.randint(5, 15)) # Reduced samples

            # Normalize data to 0-1 range using R
            X_train_normalized = X_train / R
            y_train_normalized = y_train / R
            X_test_normalized = X_test / R
            y_test_normalized = y_test / R

            print(f"Starting training cycle with {X_train_normalized.shape[0]} training samples...")
            history = model.fit(
                X_train_normalized, y_train_normalized,
                epochs=epochs,
                batch_size=500, 
                validation_data=(X_test_normalized, y_test_normalized),
                verbose=1 
            )

            print("\\nTraining cycle complete.")
            val_loss = history.history.get('val_loss', [float('inf')])[-1]
            print(f"Validation loss for this cycle: {val_loss:.4f}")

            if random.random() < 0.1:  # Show detailed info only 10% of the time
                print("\\nShowing sample prediction metrics...")
                i = 0
                raw_input_lines = X_test[i]
                true_output_lines = y_test[i]
                
                input_for_prediction = X_test_normalized[i][np.newaxis, ...]
                prediction_normalized = model.predict(input_for_prediction, verbose=0)[0]
                predicted_output_lines = prediction_normalized * R
                
                print(f"Sample input coords: {raw_input_lines[:2]}")
                print(f"True output coords: {true_output_lines[:2]}")
                print(f"Predicted coords: {predicted_output_lines[:2]}")
            
            print(f"Saving model weights to {weights_file}...")
            model.save_weights(weights_file)
            print(f"Weights saved. Continuing to next training cycle (Ctrl+C to stop).")

    except KeyboardInterrupt:
        print("\\nTraining interrupted by user. Final weights should be saved.")
    except Exception as e:
        print(f"An error occurred during training: {e}")

