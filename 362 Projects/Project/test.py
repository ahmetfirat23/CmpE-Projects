import wave

def check_wav_file(filename):
    with wave.open(filename, 'rb') as wav_file:
        # Get the sample width in bytes
        sample_width = wav_file.getsampwidth()
        # Get the number of channels
        channels = wav_file.getnchannels()
        # Get the frame rate
        frame_rate = wav_file.getframerate()
        # Get the number of frames
        n_frames = wav_file.getnframes()
        # Get the bit depth (sample width in bits)
        bit_depth = sample_width * 8

        print(f"Sample Width: {sample_width} bytes")
        print(f"Number of Channels: {channels}")
        print(f"Frame Rate: {frame_rate} Hz")
        print(f"Number of Frames: {n_frames}")
        print(f"Bit Depth: {bit_depth} bits")

        return bit_depth

filename = 'dudu.wav'
bit_depth = check_wav_file(filename)

if bit_depth == 16:
    print("16-bit samples detected.")
    # Typically .wav files with 16-bit depth use signed integers
    AmpMax = 32767
    print(f"AmpMax should be: {AmpMax}")
else:
    print(f"Bit depth of {bit_depth} detected. Please adjust the script for other bit depths.")
