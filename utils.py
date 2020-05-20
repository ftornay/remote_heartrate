from scipy import signal
import cv2
import pyramids
import preprocessing
import eulerian
import os, subprocess

def get_heartrate(filename, dir):
    """ Read video file and estimate heart rate """
    # Frequency range for Fast-Fourier Transform
    freq_min = 1
    freq_max = 1.8

    # Preprocessing phase
    hr = -1 # Valor por defecto

    fn = os.path.join(dir, filename)
    video_frames, frame_ct, fps = preprocessing.read_video(fn)

    # Build Laplacian video pyramid

    lap_video = pyramids.build_video_pyramid(video_frames)

    amplified_video_pyramid = []

    for i, video in enumerate(lap_video):
        if i == 0 or i == len(lap_video)-1:
            continue

        # Eulerian magnification with temporal FFT filtering

        result, fft, frequencies = eulerian.fft_filter(video, freq_min, freq_max, fps)
        lap_video[i] += result

        # Calculate heart rate

        hr = find_heart_rate(fft, frequencies, freq_min, freq_max)

    filename, _ = os.path.splitext(filename)
    return hr, filename

def find_heart_rate(fft, freqs, freq_min, freq_max):
    """ Calculate heart rate from FFT peaks """
    fft_maximums = []

    for i in range(fft.shape[0]):
        if freq_min <= freqs[i] <= freq_max:
            fftMap = abs(fft[i])
            fft_maximums.append(fftMap.max())
        else:
            fft_maximums.append(0)

    peaks, properties = signal.find_peaks(fft_maximums)
    max_peak = -1
    max_freq = 0

    # Find frequency with max amplitude in peaks
    for peak in peaks:
        if fft_maximums[peak] > max_freq:
            max_freq = fft_maximums[peak]
            max_peak = peak

    return freqs[max_peak] * 60

def convert_vid(path, dir, ext='mov'):
    """ Converts video to format
    corrresponding to parameter ext
    using ffmpeg """
    fn, old_ext = os.path.splitext(path)
    # Don't convert if same format
    if old_ext == ext:
        return path
    fn += '.' + ext
    src = os.path.join(dir, path)
    out = os.path.join(dir, fn)
    # -y makes sure you overwrite existing files
    cp = subprocess.run(['ffmpeg', '-y', '-i', src, out])
    if cp.returncode == 0:
        # Conversion successful!
        return fn
    else:
        return None
