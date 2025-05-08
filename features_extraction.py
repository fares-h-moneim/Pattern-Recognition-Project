import librosa
import numpy as np

def load_file(audio_path, sample_rate = 22050):
    """
    Load an audio file and return the audio time series and sampling rate.
    
    :param audio_path: Path to the audio file
    :param sr: Sampling rate
    :return: Tuple of (audio time series, sampling rate)
    """
    audio, sr = librosa.load(audio_path, sr = sample_rate)
    audio = librosa.to_mono(audio)  # Convert to mono if stereo

    return audio, sr

def extract_mfcc(audio, sample_rate = 22050, n_mfcc = 13):
    """
    Extract MFCC features from an audio signal.
    
    :param audio: Audio time series
    :param sample_rate: Sampling rate
    :param n_mfcc: Number of MFCC features to extract
    :return: MFCC features, Delta-MFCC, and Delta-Delta-MFCC
    """
    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y = audio, sr = sample_rate, n_mfcc = n_mfcc)
    mfccs_mean = np.mean(mfccs, axis=1)
    mfccs_std = np.std(mfccs, axis=1)

    # Extract Delta-MFCCs (first derivative)
    delta_mfccs = librosa.feature.delta(mfccs)
    delta_mfccs_mean = np.mean(delta_mfccs, axis=1)
    delta_mfccs_std = np.std(delta_mfccs, axis=1)

    # Extract Delta-Delta-MFCCs (second derivative)
    delta2_mfccs = librosa.feature.delta(mfccs, order=2)
    delta2_mfccs_mean = np.mean(delta2_mfccs, axis=1)
    delta2_mfccs_std = np.std(delta2_mfccs, axis=1)
    
    features = np.concatenate((
        mfccs_mean, mfccs_std, 
        delta_mfccs_mean, delta_mfccs_std, 
        delta2_mfccs_mean, delta2_mfccs_std
    ))
    return features

def extract_pitch(audio, sample_rate = 22050):
    """
    Extract pitch features from an audio signal.
    
    :param audio: Audio time series
    :param sample_rate: Sampling rate
    :return: Pitch mean and standard deviation
    """
    pitches, magnitudes = librosa.piptrack(y = audio, sr = sample_rate)

    # Compute pitch statistics
    pitch_mean = np.mean(pitches[pitches > 0])  # Average pitch
    pitch_std = np.std(pitches[pitches > 0])    # Pitch variability

    # Compute magnitude statistics
    magnitude_mean = np.mean(magnitudes[magnitudes > 0])  # Average magnitude
    magnitude_std = np.std(magnitudes[magnitudes > 0])    # Magnitude variability

    features = np.concatenate(([pitch_mean], [pitch_std], [magnitude_mean], [magnitude_std]))

    return features

if __name__ == "__main__":
    input_audio = ""

    # Load the audio file
    y, sr = load_file(input_audio)
