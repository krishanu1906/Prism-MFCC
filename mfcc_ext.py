import librosa
import librosa.display
import matplotlib.pyplot as plt
import ffmpeg

def convert_mp3_to_wav(mp3_file):
    # Generate the output WAV file name
    wav_file = mp3_file.replace('.mp3', '.wav')
    
    # Convert MP3 to WAV using ffmpeg
    ffmpeg.input(mp3_file).output(wav_file).run()
    
    return wav_file

def visualize_mfcc(audio_file):
    # Load the audio file
    y, sr = librosa.load(audio_file)

    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Display MFCCs
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfccs, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()

# Input MP3 audio file
mp3_file = "C:\\Users\\sumita\\Downloads\\Samsung\\audio.mp3"

# Convert MP3 to WAV
wav_file = convert_mp3_to_wav(mp3_file)

# Extract MFCC features and visualize
visualize_mfcc(wav_file)
