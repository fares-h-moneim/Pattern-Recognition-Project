import os
import librosa
import numpy as np
import webrtcvad
from scipy.io.wavfile import write
from noisereduce import reduce_noise


def load_audio(input_path, sample_rate = 16000):
    """
    Load audio file and convert to mono.
    
    Args:
        input_path (str): Path to the input audio file.
        sample_rate (int): Target sample rate for processing.
    
    Returns:
        audio: Loaded audio data (numpy array).
        sr: Sample rate of the loaded audio.
    """
    # Load audio file
    audio, sr = librosa.load(input_path, sr = sample_rate)
    audio = librosa.to_mono(audio)  # Convert to mono if stereo
    
    return audio, sr

def noise_reduction(input_audio, sample_rate = 16000):
    """
    Apply noise reduction to the input audio file and return the processed audio.
    
    Args:
        input_audio: Input audio data (numpy array).
        sample_rate (int): Target sample rate for processing.

        Returns:
        audio: Reduced noise audio data (numpy array).
    """
    # Apply noise reduction
    reduced_noise = reduce_noise(y = input_audio, sr = sample_rate)
    
    return reduced_noise

def silence_removal(input_audio, sample_rate = 16000):
    """
    Apply Voice Activity Detection (VAD) to the input audio file and return the processed audio.
    
    Args:
        input_audio: Input audio data (numpy array).
        sample_rate (int): Target sample rate for processing.

        Returns:
            audio: Processed audio data (numpy array).
    """
    vad = webrtcvad.Vad()
    vad.set_mode(3)  # Aggressiveness mode (0-3, 3 is the most aggressive)
    
    frame_duration = 30  # in ms
    frame_length = int(sample_rate * frame_duration / 1000)
    frames = [
        input_audio[i:i + frame_length]
        for i in range(0, len(input_audio), frame_length)
    ]
    
    # Filter out silent frames
    voiced_frames = []
    for frame in frames:
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')

        # Convert frame to 16-bit PCM format
        frame_int16 = (frame * 32767).astype(np.int16)  # Scale to 16-bit PCM
        frame_bytes = frame_int16.tobytes()  # Convert to bytes
        
        # Check if the frame contains speech
        is_speech = vad.is_speech(frame_bytes, sample_rate)
        if is_speech:
            voiced_frames.extend(frame)
    
    # Return the processed audio
    processed_audio = np.array(voiced_frames, dtype=np.float32)
    return processed_audio


def write_audio(input_audio, output_path, sample_rate = 16000):
    write(output_path, sample_rate, (input_audio * 32767).astype(np.int16))


if __name__ == "__main__":
    input_path = ""
    output_path = ""
    
    input_audio, sample_rate = load_audio(input_path)

    audio_without_noise = noise_reduction(input_audio, sample_rate)
