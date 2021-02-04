import argparse
import os
import pyaudio
import time
import wave
from scipy import signal
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
from io import BytesIO
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument("--num-samples",type=int,required=True)
parser.add_argument("--output",type=str,required=True)
args=parser.parse_args()
f_l=int(16000*0.04)
f_s=int(16000*0.02)
num_mel_bins=40
sampling_rate=16000
lower_frequency=20
upper_frequency=4000
sampling_ratio=int(48000/16000)
max_=32768
min_=-32768
num_iter=args.num_samples
resolution=pyaudio.paInt16
samp_rate=48000
chunk=4800
rec_sec=1
dev_index=0
wav_out_file=f'/home/pi/WORK_DIR/{args.output}'
num_spectrogram_bins = 321
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(num_mel_bins,num_spectrogram_bins,sampling_rate,lower_frequency,upper_frequency)
buffer=BytesIO()
audiomain=pyaudio.PyAudio()
stream=audiomain.open(format=resolution, rate=samp_rate, channels=1, input_device_index=dev_index, input=True, frames_per_buffer=chunk)
stream.stop_stream()

subprocess.Popen(["sudo","sh","-c","echo 1 > /sys/devices/system/cpu/cpufreq/policy0/stats/reset"]).wait()   
for tempo in range(num_iter):
    start=time.time()
    
    start_rec=time.time()
    stream.start_stream()
    subprocess.Popen(["sudo","sh","-c",
                 "echo powersave> /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"])
    buffer.seek(0)
    for i in range(int((samp_rate/chunk)*rec_sec)):
        if i==9:
            subprocess.Popen(["sudo","/bin/sh","-c",
                                "echo performance > /sys/devices/system/cpu/cpufreq/policy0/scaling_governor"])

        buffer.write(stream.read(chunk))
        
    stream.stop_stream()
    end_rec=time.time()
    
    #RESAMPLE
    audio=signal.resample_poly(np.frombuffer(buffer.getvalue(),dtype=np.int16),1,sampling_ratio)
    audio=audio.astype(np.int16)
    
    #STFT
    tf_audio=tf.convert_to_tensor(audio, dtype=tf.float32)
    tf_audio=(2*(tf_audio - min_))/(max_ - min_) -1
    stft=tf.abs(tf.signal.stft(tf_audio, frame_length=f_l, frame_step=f_s, fft_length=f_l))
    #end_stft=time.time()
    
    #MFCCS
    mel_spectrogram = tf.tensordot(stft,linear_to_mel_weight_matrix,1)
    mel_spectrogram.set_shape(stft.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10]
    
    #SAVE
    conversion=tf.io.serialize_tensor(mfccs)
    tf.io.write_file(f"/home/pi/WORK_DIR/{args.output}/mfccs{tempo}.bin",conversion)
    
    end=time.time()
    
    print(f'{end-start:.3f}')


subprocess.run(["cat","/sys/devices/system/cpu/cpufreq/policy0/stats/time_in_state"])
stream.close()
audiomain.terminate()



