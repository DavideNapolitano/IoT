import argparse
import os
import pyaudio
import time
import wave
from scipy.io import wavfile
from scipy import signal
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime

parser=argparse.ArgumentParser()
parser.add_argument("--input",type=str,required=True)
parser.add_argument("--output",type=str,required=True)
args=parser.parse_args()

filename = f'/home/pi/WORK_DIR/{args.output}'
input_=os.listdir(f'/home/pi/WORK_DIR/{args.input}')

for el in input_:
    if el=="samples.csv":
        data=pd.read_csv(f'/home/pi/WORK_DIR/{args.input}/{el}', header=None)

with tf.io.TFRecordWriter(filename) as writer:
     for i in range(data.shape[0]):  
        l=data[0][i].split("/")
        t=data[1][i].split(":")
        d=datetime(int(l[2]),int(l[1]),int(l[0]),int(t[0]),int(t[1]),int(t[2]))
        ts=d.timestamp()
        
        dat=tf.cast(int(ts),tf.int32).numpy()
        t=data[2][i]
        h=data[3][i]
        a=tf.io.read_file(f'/home/pi/WORK_DIR/{args.input}/{data[4][i]}')
        
        date = tf.train.Feature(int64_list=tf.train.Int64List(value=[dat]))
        temp = tf.train.Feature(int64_list=tf.train.Int64List(value=[t]))
        hum = tf.train.Feature(int64_list=tf.train.Int64List(value=[h]))
        audio= tf.train.Feature(bytes_list=tf.train.BytesList(value=[a.numpy()]))
        
        mapping = {'datetime': date, 'temperature' : temp, 'humidity' : hum,
                    'audio' : audio
                   }
        example = tf.train.Example(features=tf.train.Features(feature=mapping))
                               
        writer.write(example.SerializeToString())

size=os.path.getsize('/home/pi/WORK_DIR/fusion.tfrecord')
print(f'Size file: {size}')
