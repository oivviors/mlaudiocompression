import librosa
import numpy as np
from scipy.stats import entropy
from scipy import stats
import time
import platform
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.models import Model
import argparse


def configure_gpu():
    """
    Configure GPU usage for TensorFlow, specifically for macOS.
    Returns True if GPU is available and configured, False otherwise.
    """
    # Check if running on macOS
    if platform.system() == "Darwin":
        try:
            # Check if Metal plugin is available
            if tf.config.list_physical_devices("GPU"):
                # Set memory growth to avoid taking all GPU memory
                for gpu in tf.config.list_physical_devices("GPU"):
                    tf.config.experimental.set_memory_growth(gpu, True)

                # Set mixed precision policy for better performance
                tf.keras.mixed_precision.set_global_policy("mixed_float16")

                print("Metal GPU detected and configured for TensorFlow")
                print(f"TensorFlow version: {tf.__version__}")
                print(f"GPU devices: {tf.config.list_physical_devices('GPU')}")
                return True
            else:
                print("No Metal GPU detected. Using CPU for TensorFlow operations.")
                return False
        except Exception as e:
            print(f"Error configuring GPU: {e}")
            return False
    else:
        # For non-macOS systems, let TensorFlow handle GPU detection
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU detected and configured: {gpus}")
                return True
            except RuntimeError as e:
                print(f"Error configuring GPU: {e}")
                return False
        else:
            print("No GPU detected. Using CPU for TensorFlow operations.")
            return False


def calculate_entropy(array):
    # Flatten the array to 1D
    array = np.asarray(array).flatten()

    # Calculate histogram (this works with negative values)
    # For int16, we use bins from -32768 to 32767
    hist, _ = np.histogram(array, bins=np.arange(-32768, 32768, 1))

    # Normalize to get probabilities
    probabilities = hist[hist > 0] / len(array)

    # Calculate entropy
    return entropy(probabilities, base=2)


def load_wav_file(file_path):
    """
    Load a WAV file and return it as a NumPy array.

    Args:
        file_path (str): Path to the WAV file

    Returns:
        tuple: (audio_array, sample_rate)
            - audio_array: NumPy array containing the audio data (as float32)
            - sample_rate: The sample rate of the audio file
    """
    try:
        # Load with librosa for robust WAV handling
        audio_array, sample_rate = librosa.load(file_path, sr=None)
        return audio_array, sample_rate
    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None, None


def load_wav_file_as_int16(file_path):
    """
    Load a WAV file and return it as a NumPy array of int16.
    """
    audio_array, sample_rate = load_wav_file(file_path)
    return audio_array.astype(np.int16) * 2**15


def calculate_sample_differences(array):
    """
    Calculate the difference between consecutive samples in an audio array.
    For the first sample, output its actual value.

    Args:
        array (numpy.ndarray): Input audio array

    Returns:
        numpy.ndarray: Array of differences between consecutive samples
    """
    # Create an array to store the differences
    differences = np.zeros_like(array)

    # First sample is output as is
    differences[0] = array[0]

    # For all other samples, calculate the difference from the previous sample
    differences[1:] = array[1:] - array[:-1]

    return differences


def linear_prediction(array, n=4):
    """
    Perform linear prediction using n previous samples to predict the next sample.
    Uses linear regression to fit a line to the previous n samples and extrapolate.
    The first n samples are output as is.

    Args:
        array (numpy.ndarray): Input audio array
        n (int): Number of previous samples to use for prediction

    Returns:
        numpy.ndarray: Array of prediction errors (actual - predicted)
    """
    prediction_errors = np.zeros_like(array)

    prediction_errors[:n] = array[:n]

    # For all other samples, calculate the prediction error
    for i in range(n, len(array)):
        # Get the n previous samples
        previous_samples = array[i - n : i]

        # Create x values (indices) for the previous samples
        x = np.arange(n)

        # Perform linear regression
        slope, intercept, _, _, _ = stats.linregress(x, previous_samples)

        # Predict the next sample by extrapolating the line
        predicted = slope * n + intercept

        # Calculate the prediction error (actual - predicted)
        prediction_errors[i] = array[i] - predicted

    return prediction_errors


def polynomial_prediction(
    array: np.ndarray[np.float32], n=4, degree=2
) -> np.ndarray[np.int16]:
    """
    Perform polynomial prediction using n previous samples to predict the next sample.
    Uses polynomial fitting to fit a curve to the previous n samples and extrapolate.
    The first n samples are output as is.

    Args:
        array (numpy.ndarray[np.float32]): Input audio array as float32
        n (int): Number of previous samples to use for prediction
        degree (int): Degree of the polynomial to fit (default: 2)

    Returns:
        numpy.ndarray: Array of prediction errors (actual - predicted)
    """
    # Create an array to store the prediction errors
    prediction_errors = np.zeros_like(array)

    # First n samples are output as is
    prediction_errors[:n] = array[:n]

    # Ensure degree is not too high relative to n
    if degree >= n:
        print(
            f"Warning: Degree {degree} is too high for {n} samples. Reducing to {n - 1}."
        )
        degree = n - 1

    # For all other samples, calculate the prediction error
    for i in range(n, len(array)):
        # Get the n previous samples
        previous_samples = array[i - n : i]

        # Create x values (indices) for the previous samples
        x = np.arange(n)

        # Check if we have enough variation in the data

        try:
            # Fit a polynomial of specified degree with rcond parameter to improve conditioning
            coefficients = np.polyfit(x, previous_samples, degree, rcond=1e-10)

            # Predict the next sample by evaluating the polynomial at x=n
            predicted = np.polyval(coefficients, n)
        except np.RankWarning:
            # If we still get a warning, fall back to linear prediction
            print(
                "Warning: RankWarning encountered. Falling back to linear prediction."
            )
            slope, intercept, _, _, _ = stats.linregress(x, previous_samples)
            predicted = slope * n + intercept

        # Calculate the prediction error (actual - predicted)
        prediction_errors[i] = array[i] - predicted

    # Normalize prediction errors to [-1, 1] range
    max_abs_error = np.max(np.abs(prediction_errors))
    if max_abs_error > 0:
        prediction_errors = prediction_errors / max_abs_error

    # Convert to int16 range (-32768 to 32767)
    # Use 0.99 to avoid potential overflow
    prediction_errors = (prediction_errors * 0.999 * 32767).astype(np.int16)

    return prediction_errors


def simple_linear_prediction(array, n=2):
    """
    Perform linear prediction using n previous samples to predict the next sample.

    But instead of using linear regression we just average the differences between the previous n consecutive samples.

    Args:
        array (numpy.ndarray): Input audio array
        n (int): Number of previous samples to use for prediction

    Returns:
        numpy.ndarray: Array of prediction errors (actual - predicted)
    """
    # Create an array to store the prediction errors
    prediction_errors = np.zeros_like(array)

    # First n samples are output as is
    prediction_errors[:n] = array[:n]

    # For all other samples, calculate the prediction error
    for i in range(n, len(array)):
        # Get the n previous samples
        previous_samples = array[i - n : i]

        # Calculate differences between consecutive samples
        differences = np.diff(previous_samples)

        # Average the differences to estimate the next difference
        avg_difference = np.mean(differences)

        # Predict the next sample by adding the average difference to the last sample
        predicted = previous_samples[-1] + avg_difference

        # Calculate the prediction error (actual - predicted)
        prediction_errors[i] = array[i] - predicted

    return prediction_errors


def histogram_cli_representation(
    array,
    width=100,
    height=20,
    min_value=-32768,
    max_value=32767,
    compute_min_max=False,
    logarithmic=True,
):
    """
    Convert an array into a CLI-friendly histogram representation
    and return as a string.

    Args:
        array (numpy.ndarray): Input array to visualize
        width (int): Width of the histogram in characters
        height (int): Height of the histogram in lines
        min_value (int): Minimum value for the x-axis
        max_value (int): Maximum value for the x-axis

    Returns:
        str: CLI-friendly histogram representation
    """
    # Calculate histogram bins
    if compute_min_max:
        min_value = np.min(array)
        max_value = np.max(array)
    hist, bin_edges = np.histogram(array, bins=width, range=(min_value, max_value))

    # Normalize histogram to fit within the specified height
    max_count = np.max(hist)
    if max_count == 0:
        return "Empty histogram (no data)"

    # Apply logarithmic scaling if requested
    if logarithmic:
        # Add a small constant to avoid log(0)
        hist = np.log10(hist + 1)
        max_count = np.max(hist)

    # Create the histogram visualization
    result = []

    # Add y-axis labels and bars
    for i in range(height, 0, -1):
        # Calculate threshold for this row
        threshold = max_count * (i / height)

        # Create the row
        row = []
        for count in hist:
            if count >= threshold:
                row.append("█")
            else:
                row.append(" ")

        # Add the row to the result
        result.append("".join(row))

    # Add x-axis labels
    x_axis = []
    for i in range(0, width, width // 10):
        x_axis.append(f"{bin_edges[i]:6.2f}")

    # Join the x-axis labels
    x_axis_str = " ".join(x_axis)

    # Add the x-axis to the result
    result.append("-" * width)
    result.append(x_axis_str)

    # Add a note about logarithmic scaling if used
    if logarithmic:
        result.append("\n(Logarithmic scale)")

    # Return the complete histogram
    return "\n".join(result)


def prepare_training_data(wav_files, sequence_length):
    """
    Prepare training data from a set of WAV files.

    Args:
        wav_files (list): List of paths to WAV files
        sequence_length (int): Number of previous samples to use for prediction

    Returns:
        tuple: (X, y) where X is the input sequences and y is the target values
    """
    all_sequences = []
    all_targets = []

    for wav_file in wav_files:
        # Load the audio file
        audio_data, sample_rate = load_wav_file(wav_file)

        # Map to float32 in range [-1, 1]

        # Create sequences of previous samples and their targets
        for i in range(len(audio_data) - sequence_length):
            sequence = audio_data[i : i + sequence_length]
            target = audio_data[i + sequence_length]
            if i % 100 == 0:  # For now
                all_sequences.append(sequence)
                all_targets.append(target)

    # Convert to numpy arrays
    X = np.array(all_sequences, dtype=np.float32)
    y = np.array(all_targets, dtype=np.float32)
    return X, y


def train_prediction_model(
    wav_files,
    model_path,
    sequence_length,
    epochs=50,
    batch_size=32,
):
    """
    Train a neural network to predict the next audio sample.

    Args:
        wav_files (list): List of paths to WAV files for training
        model_path (str): Path to save the trained model
        sequence_length (int): Number of previous samples to use for prediction
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training

    Returns:
        tuple: (model, history) where model is the trained model and history contains training metrics
    """
    gpu_available = configure_gpu()
    print(f"Using {'GPU' if gpu_available else 'CPU'} for training")

    # Prepare training data
    X, y = prepare_training_data(wav_files, sequence_length)

    # Split into training and validation sets
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"X_train shape: {X_train.shape}")

    # Flatten the input for dense layers
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_val_flat = X_val.reshape(X_val.shape[0], -1)

    # Create the model using the recommended approach with Input layer
    # Define the input layer
    inputs = Input(shape=(X_train_flat.shape[1],))

    # Add the first dense layer
    x = Dense(32)(inputs)

    # Add activation layer - using tanh which is better for audio data in [-1, 1] range
    x = tf.keras.layers.Activation("tanh")(x)

    # Add the output layer
    outputs = Dense(1)(x)

    # Create the model
    model = Model(inputs=inputs, outputs=outputs)

    # Configure SGD optimizer for audio data
    # SGD often provides better generalization than Adam for certain types of data
    optimizer = SGD(
        learning_rate=0.01,  # Learning rate for SGD
        momentum=0.9,  # Momentum helps SGD navigate local minima
        nesterov=True,  # Nesterov momentum can provide better convergence
    )

    if gpu_available:
        optimizer = tf.keras.mixed_precision.LossScaleOptimizer(optimizer)

    # Compile the model with MSE loss which is appropriate for audio prediction
    model.compile(optimizer=optimizer, loss="mse")

    # Callbacks for training
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, monitor="val_loss"),
    ]

    # Train the model
    print(f"Training model on {len(wav_files)} files...")

    # Use tf.data.Dataset for better performance
    train_dataset = (
        tf.data.Dataset.from_tensor_slices((X_train_flat, y_train))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_dataset = (
        tf.data.Dataset.from_tensor_slices((X_val_flat, y_val))
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    # Print training summary
    print("\nTraining Summary:")
    print(f"Final training loss: {history.history['loss'][-1]:.6f}")
    print(f"Final validation loss: {history.history['val_loss'][-1]:.6f}")
    print(f"Best validation loss: {min(history.history['val_loss']):.6f}")
    print(f"Epochs trained: {len(history.history['loss'])}")

    return model, history


def predict_with_model(model_path, audio_data, sequence_length=10):
    """
    Use a trained model to predict the next sample and calculate prediction errors.

    Args:
        model_path: Path to the trained model file
        audio_array (numpy.ndarray): Input audio array
        sequence_length (int): Number of previous samples to use for prediction

    Returns:
        numpy.ndarray: Array of prediction errors (actual - predicted)
    """
    gpu_available = configure_gpu()
    print(f"Using {'GPU' if gpu_available else 'CPU'} for inference")

    # Load the trained model
    model = load_model(model_path)

    # Create an array to store the prediction errors
    prediction_errors = np.zeros_like(audio_data, dtype=np.float32)

    # First sequence_length samples are output as is
    prediction_errors[:sequence_length] = audio_data[:sequence_length]

    # For all other samples, calculate the prediction error
    count = 0

    start_time = time.time()

    # Process in batches for better GPU utilization
    batch_size = 32
    stop_at = 44100 * 2 / batch_size
    for i in range(sequence_length, len(audio_data), batch_size):
        end_idx = min(i + batch_size, len(audio_data))
        batch_sequences = []

        for j in range(i, end_idx):
            sequence = audio_data[j - sequence_length : j]
            batch_sequences.append(sequence)

        # Reshape for model input
        batch_sequences = np.array(batch_sequences).reshape(-1, sequence_length, 1)

        # Predict the next samples
        predicted = model.predict(batch_sequences, verbose=0)

        # Calculate the prediction errors
        for j, pred in enumerate(predicted):
            idx = i + j
            if idx < len(audio_data):
                error = audio_data[idx] - pred[0]
                prediction_errors[idx] = error
                # if error == 0:
                #     print(f"Error is 0 at {idx}")
                # else:
                #     print(f"Error is NON ZERO {error} at {idx}")
                count += 1

                if count % 1000 == 0:
                    print(
                        f"Predicted/actual: {int(pred[0] * 2**15):6d}, {int(audio_data[idx] * 2**15):6d} {int(error * 2**15):6d} ({idx}/{len(audio_data)} percent: {idx / len(audio_data) * 100:.2f}% )"
                    )

                # if count > stop_at:
                #    break

        # if count > stop_at:
        #    break

    end_time = time.time()
    seconds_per_sample = (end_time - start_time) / count
    time_to_predict_a_second = seconds_per_sample * 44100
    print(f"Prediction took {end_time - start_time:.4f} seconds")
    print(f"Time to predict a second: {time_to_predict_a_second:.4f} seconds")

    # Ensure the entire array is int16
    prediction_errors = (prediction_errors * 2**15).astype(np.int16)
    return prediction_errors


def preliminary_work():
    random_values = np.random.randint(
        -32768, 32767, size=44100 * 10
    )  # Uniformly random 16-bit integers

    print(f"Entropy of random_values: {calculate_entropy(random_values)}")

    file_path = "audio/iphone_10secs.wav"  # You can change this to any WAV file in your directory

    # Load with scipy (preserves original data type)
    audio_array_original, sample_rate_original = load_wav_file(
        file_path, preserve_dtype=True
    )
    audio_array_10secs = audio_array_original[: 44100 * 10]

    print(f"Entropy of audio_array_original: {calculate_entropy(audio_array_10secs)}")

    # Compare both prediction methods with timing
    print("\nUsing linear regression prediction:")
    start_time = time.time()
    prediction_errors = linear_prediction(audio_array_10secs, n=2)
    regression_time = time.time() - start_time
    print(f"Linear regression prediction took {regression_time:.4f} seconds")
    print(histogram_cli_representation(prediction_errors))

    print("\nUsing simple difference-based prediction:")
    start_time = time.time()
    simple_prediction_errors = simple_linear_prediction(audio_array_10secs, n=2)
    simple_time = time.time() - start_time
    print(f"Simple prediction took {simple_time:.4f} seconds")
    print(histogram_cli_representation(simple_prediction_errors))

    # Compare entropies and timing
    print(
        f"\nEntropy of linear regression prediction errors: {calculate_entropy(prediction_errors):.2f}"
    )
    print(
        f"Entropy of simple prediction errors: {calculate_entropy(simple_prediction_errors):.2f}"
    )
    print(f"Speed improvement: {regression_time / simple_time:.2f}x faster")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Audio prediction model training and inference"
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument(
        "--predict", action="store_true", help="Run prediction on an audio file"
    )

    parser.add_argument(
        "--polynomial", action="store_true", help="Run polynomial prediction"
    )

    parser.add_argument("--prelim", action="store_true", help="Show preliminary work")

    args = parser.parse_args()

    # If no mode is specified, show help
    if not args.train and not args.predict and not args.prelim and not args.polynomial:
        parser.print_help()
        exit(1)

    if args.prelim:
        preliminary_work()
        exit(1)

    if args.polynomial:
        print("Using polynomial prediction")
        audio_array, sample_rate = load_wav_file("audio/iphone_another_1s.wav")
        prediction_errors = polynomial_prediction(audio_array, n=100, degree=10)
        print(histogram_cli_representation(prediction_errors, logarithmic=True))
        print(f"Entropy of prediction errors: {calculate_entropy(prediction_errors)}")
        exit(1)

    # Settings (kept in code as requested)
    sequence_length = 5
    model_path = "audio_prediction_model2.keras"

    # Training mode
    if args.train:
        wav_files = ["audio/iphone_10secs.wav"]
        print(f"Training model with sequence length {sequence_length}")
        print(f"Using WAV files: {wav_files}")
        model, history = train_prediction_model(
            wav_files, model_path, sequence_length=sequence_length
        )
        print("Training completed. Model saved to:", model_path)

    # Prediction mode
    if args.predict:
        audio_file = "audio/iphone_another_1s.wav"
        print(
            f"Running prediction on {audio_file} with sequence length {sequence_length}"
        )
        audio_array, sample_rate = load_wav_file(audio_file)
        prediction_errors = predict_with_model(
            model_path, audio_array, sequence_length=sequence_length
        )
        print(f"Entropy of prediction errors: {calculate_entropy(prediction_errors)}")
        print(histogram_cli_representation(prediction_errors, compute_min_max=False))
