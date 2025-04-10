import librosa


def load_wav_file(file_path):
    """
    Load a WAV file and return it as a NumPy array.

    Args:
        file_path (str): Path to the WAV file

    Returns:
        tuple: (audio_array, sample_rate)
            - audio_array: NumPy array containing the audio data
            - sample_rate: The sample rate of the audio file
    """
    try:
        # Load the audio file
        audio_array, sample_rate = librosa.load(file_path, sr=None)

        print(f"Successfully loaded {file_path}")
        print(f"Audio shape: {audio_array.shape}")
        print(f"Sample rate: {sample_rate} Hz")
        print(f"Duration: {len(audio_array) / sample_rate:.2f} seconds")

        return audio_array, sample_rate

    except Exception as e:
        print(f"Error loading file: {str(e)}")
        return None, None


if __name__ == "__main__":
    # Example usage
    file_path = "audio/bill_melson_050117.wav"  # You can change this to any WAV file in your directory
    audio_array, sample_rate = load_wav_file(file_path)

    # if audio_array is not None:
    #     # You can now work with the audio_array as a NumPy array
    #     # For example, print some basic statistics
    #     print("\nAudio statistics:")
    #     print(f"Mean amplitude: {np.mean(audio_array):.4f}")
    #     print(f"Max amplitude: {np.max(np.abs(audio_array)):.4f}")
    #     print(f"Min amplitude: {np.min(audio_array):.4f}")
