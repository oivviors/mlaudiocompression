import librosa
import numpy as np
from scipy.io import wavfile
from scipy.stats import entropy
from scipy import stats
import time
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


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


def load_wav_file(file_path, preserve_dtype=True):
    """
    Load a WAV file and return it as a NumPy array.

    Args:
        file_path (str): Path to the WAV file
        preserve_dtype (bool): If True, use scipy to load the file and preserve the original data type

    Returns:
        tuple: (audio_array, sample_rate)
            - audio_array: NumPy array containing the audio data
            - sample_rate: The sample rate of the audio file
    """
    try:
        if preserve_dtype:
            # Use scipy to load the file and preserve the original data type
            sample_rate, audio_array = wavfile.read(file_path)
            print("Loaded with scipy (preserving original data type)")
        else:
            # Load the audio file with librosa (converts to float32)
            audio_array, sample_rate = librosa.load(file_path, sr=None)
            print("Loaded with librosa (converted to float32)")

        print(f"Successfully loaded {file_path}")
        print(f"Audio shape: {audio_array.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_array) / sample_rate:.2f} seconds")
        print(f"Data type: {audio_array.dtype}")

        return audio_array, sample_rate

    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None, None


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
    # Create an array to store the prediction errors
    prediction_errors = np.zeros_like(array)

    # First n samples are output as is
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


def prepare_training_data(wav_files, sequence_length=10):
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
        sample_rate, audio_data = wavfile.read(wav_file)

        # Keep as integers, just ensure int32 type for calculations
        audio_data = audio_data.astype(np.int16)

        # breakpoint()
        count = 0
        # Create sequences of previous samples and their targets
        for i in range(len(audio_data) - sequence_length):
            sequence = audio_data[i : i + sequence_length]
            target = audio_data[i + sequence_length]

            all_sequences.append(sequence)
            all_targets.append(target)
            count += 1

            # For now bail early
            if count > 100000:
                break

    # Convert to numpy arrays
    X = np.array(all_sequences, dtype=np.int16)
    y = np.array(all_targets, dtype=np.int16)

    # Reshape for LSTM input (samples, time steps, features)
    X = X.reshape((X.shape[0], X.shape[1], 1))

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
    # Prepare training data
    X, y = prepare_training_data(wav_files, sequence_length)

    # Split into training and validation sets
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"X_train shape: {X_train.shape}")

    # Create the model
    model = Sequential(
        [
            LSTM(64, input_shape=(sequence_length, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1, activation="linear"),  # Linear activation for integer output
        ]
    )

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")

    # Callbacks for training
    callbacks = [
        ModelCheckpoint(model_path, save_best_only=True, monitor="val_loss"),
        EarlyStopping(patience=5, monitor="val_loss"),
    ]

    # Train the model
    print(f"Training model on {len(wav_files)} files...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
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


def predict_with_model(model_path, audio_array, sequence_length=10):
    """
    Use a trained model to predict the next sample and calculate prediction errors.

    Args:
        model_path: Path to the trained model file
        audio_array (numpy.ndarray): Input audio array
        sequence_length (int): Number of previous samples to use for prediction

    Returns:
        numpy.ndarray: Array of prediction errors (actual - predicted)
    """
    # Load the trained model
    model = load_model(model_path)

    # Keep as int16 to match training data
    audio_data = audio_array.astype(np.int16)

    # Create an array to store the prediction errors
    prediction_errors = np.zeros_like(audio_array, dtype=np.int16)

    # First sequence_length samples are output as is
    prediction_errors[:sequence_length] = audio_array[:sequence_length]

    # For all other samples, calculate the prediction error
    count = 0
    start_time = time.time()
    for i in range(sequence_length, len(audio_array)):
        # print(f"Predicting sample {i} of {len(audio_array)}")
        # Get the sequence of previous samples
        sequence = audio_data[i - sequence_length : i]

        # Reshape for model input
        sequence = sequence.reshape((1, sequence_length, 1))
        print(sequence)

        # Predict the next sample
        predicted = model.predict(sequence, verbose=0)[0, 0]

        # Round to nearest integer and convert to int16
        predicted = np.round(predicted).astype(np.int16)

        # Calculate the prediction error (actual - predicted)
        error = audio_array[i] - predicted
        print(f"Predicted/actual: {predicted:6d}, {audio_array[i]:6d} {error:6d}")
        prediction_errors[i] = error
        count += 1
        if count > 1000:
            break
    end_time = time.time()
    seconds_per_sample = (end_time - start_time) / count
    time_to_predict_a_second = seconds_per_sample * 44100
    print(f"Prediction took {end_time - start_time:.4f} seconds")
    print(f"Time to predict a second: {time_to_predict_a_second:.4f} seconds")
    return prediction_errors


def call_train_prediction_model(sequence_length):
    model_path = "audio_prediction_model.keras"
    wav_files = ["audio/iphone_rest_of_the_file.wav"]
    model, history = train_prediction_model(
        wav_files, model_path, sequence_length=sequence_length
    )


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
    """
    # Example of using the neural network prediction
    # Uncomment and modify these lines to use the neural network
    """
    sequence_length = 5

    call_train_prediction_model(sequence_length=sequence_length)
    exit()
    model_path = "audio_prediction_model2.keras"

    # audio_file = "audio/iphone_rest_of_the_file.wav"
    audio_file = "audio/iphone_very_short.wav"
    audio_array, sample_rate = load_wav_file(audio_file, preserve_dtype=True)
    prediction_errors = predict_with_model(
        model_path, audio_array, sequence_length=sequence_length
    )
    print(f"Entropy of prediction errors: {calculate_entropy(prediction_errors)}")
    print(histogram_cli_representation(prediction_errors, compute_min_max=True))
