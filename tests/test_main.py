import pathlib
import sys
import numpy as np

# Add the parent directory to the Python path to allow importing from the main module
sys.path.append(str(pathlib.Path(__file__).parent.parent))

# Import the function to test
from main import load_wav_file
from main import prepare_training_data
from main import train_prediction_model
from main import predict_with_model


# Get the directory where the test file is located
def get_test_dir():
    """Get the directory where the current test file is located."""
    return pathlib.Path(__file__).parent.absolute()


def get_relative_path(relative_path):
    """Convert a path relative to the test file location to an absolute path."""
    return get_test_dir() / relative_path


# Define the path to the test audio file
filename_very_short = get_relative_path("../audio/iphone_very_short.wav")
filename_very_short2 = get_relative_path("../audio/iphone_another_very_short.wav")


def test_load_wav_file():
    """Test the load_wav_file function."""
    audio_array, sample_rate = load_wav_file(filename_very_short)

    assert len(audio_array) == 665

    # Assert that all values are between -1 and 1 (librosa's output range)
    assert np.all(audio_array >= -1.0)
    assert np.all(audio_array <= 1.0)


def test_prepare_training_data():
    """Test the prepare_training_data function."""
    wav_files = [filename_very_short]
    X, y = prepare_training_data(wav_files)

    assert X.shape == (655, 10)
    assert y.shape == (655,)


def test_train_prediction_model():
    """Test the train_prediction_model function."""
    wav_files = [filename_very_short]
    model, history = train_prediction_model(wav_files, "test_model.keras", 10)
    # assert model is not None
    # assert history is not None


def test_train_and_predict_audio():
    """Test the trail_prediction_model and the predict_with_model function."""

    model_path = "test_model.keras"
    model, history = train_prediction_model([filename_very_short], model_path, 10)
    very_short2, sample_rate = load_wav_file(filename_very_short2)
    predict_with_model(model_path, very_short2)
