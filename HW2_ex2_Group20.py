import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
import zlib
from collections import Counter
from scipy import signal

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version name')
args = parser.parse_args()

mfcc=True
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#3.1
zip_path = tf.keras.utils.get_file(
    origin='http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip',
    fname='mini_speech_commands.zip',
    extract=True, cache_dir='.', cache_subdir='data')
data_dir= os.path.join('.','data','mini_speech_commands')
filename=tf.io.gfile.glob(str(data_dir)+'/*/*')

train_d=pd.read_csv("kws_train_split.txt")
val_d=pd.read_csv("kws_val_split.txt")
test_d=pd.read_csv("kws_test_split.txt")
LABELS=np.array(tf.io.gfile.listdir(str(data_dir)))
LABELS=LABELS[LABELS!='README.md']
train_data=train_d.values.squeeze()
val_data=val_d.values.squeeze()
test_data=test_d.values.squeeze()
print("DATA DOWNLOADED")

size_in=32

def my_nump_func(x):
    #print(f"PRE RESAMPLE: {len(x)}")
    res_np_audio=signal.resample_poly(x, 1, 2)
    #print(f"POT RESAMPLE: {len(res_np_audio)}")
    return res_np_audio.astype(np.float32)

class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step, num_mel_bins=None, lower_frequency=None,
                 upper_frequency=None, num_coefficients=None, mfcc=False):
        tf.config.experimental_run_functions_eagerly(True)
        self.labels=labels
        self.sampling_rate=sampling_rate
        self.frame_length=frame_length
        self.frame_step=frame_step
        self.num_mel_bins=num_mel_bins
        self.lower_frequency=lower_frequency
        self.upper_frequency=upper_frequency
        self.num_spectrogram_bins=num_coefficients
        
        if mfcc is True:
            self.preprocess=self.preprocess_with_mfcc
        else:
            self.preprocess=self.preprocess_with_stft
    
    def read(self, file_path):
        parts=tf.strings.split(file_path, os.path.sep)
        label=parts[-2]
        label_id=tf.argmax(label==self.labels)
        audio_binary=tf.io.read_file(file_path)
        audio,_=tf.audio.decode_wav(audio_binary)
        audio=tf.squeeze(audio,axis=1)
        return audio,label_id
    

    def pad(self,audio):
        #print(audio.shape)
        #print(audio)
        zero_padding=tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio=tf.concat([audio,zero_padding],0)
        audio.set_shape([self.sampling_rate])
        #print(audio)
        if args.version=="c":
            print("DOWNSAMPLE")
            audio=tf.numpy_function(my_nump_func,[audio],tf.float32)
            #audio_res=tf.convert_to_tensor(audio_res)
            audio.set_shape([8000])
            #print(audio_res)
        return audio
    
    def get_spectrogram(self, audio):
        spectrogram=tf.abs(tf.signal.stft(audio, frame_length=self.frame_length, frame_step=self.frame_step, fft_length=self.frame_length))

        return spectrogram
    
    def get_mfcc(self, spectrogram):
        linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(self.num_mel_bins,self.num_spectrogram_bins,
                                                                            self.sampling_rate,self.lower_frequency,self.upper_frequency)
        mel_spectrogram = tf.tensordot(spectrogram,linear_to_mel_weight_matrix,1)
        mel_spectrogram.set_shape(spectrogram.shape[:-1].concatenate(linear_to_mel_weight_matrix.shape[-1:]))
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)[:, :10] #10

        return mfccs

    def preprocess_with_stft(self, file_path):
        print("STFT")
        audio,label=self.read(file_path)
        audio=self.pad(audio)
        spectrogram=self.get_spectrogram(audio)
        spectrogram=tf.expand_dims(spectrogram,-1)
        spectrogram=tf.image.resize(spectrogram, [size_in, size_in])

        return spectrogram, label
    
    def preprocess_with_mfcc(self, file_path):
        print("MFCC")
        audio,label=self.read(file_path)
        audio=self.pad(audio)
        stft=self.get_spectrogram(audio)
        mfcc=self.get_mfcc(stft)
        mfccs=tf.expand_dims(mfcc,-1)
        return mfccs, label

    def make_dataset(self, files, train): #files=list file path
        ds=tf.data.Dataset.from_tensor_slices(files)
        ds=ds.map(self.preprocess,num_parallel_calls=1)
        ds=ds.batch(32)
        ds=ds.cache()
        if train is True:
            ds=ds.shuffle(100,reshuffle_each_iteration=True)
            
        return ds
    
tf.config.experimental_run_functions_eagerly(True)
if mfcc:
    if args.version=="c":
        print("HYPER FOR DOWNSAMPLE")
        generator=SignalGenerator(LABELS,16000,320,160,40,20,4000,161,mfcc=mfcc)
    else:
        generator=SignalGenerator(LABELS,16000,640,320,40,20,4000,321,mfcc=mfcc)
else:
  generator=SignalGenerator(LABELS,8000,256,128,mfcc=mfcc) #stft
  
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
tf.data.experimental.save(test_ds, './th_test')
print("DATASET GENERATED")

if mfcc:
  strides=[2,1]
  in_shape=(49,10,1)
else:
  strides=[2,2]
  in_shape=(size_in,size_in,1)
  
if args.version=="a" or args.version=="b":
    model = keras.Sequential([
          keras.layers.Conv2D(filters=96, kernel_size=[3, 3], strides=strides, use_bias=False,input_shape=in_shape),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
          keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
          keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(units=8),
      ])
    NUM_EPOCHS=20
else:
    print("MODEL FOR VERSION C")
    model = keras.Sequential([
          keras.layers.Conv2D(filters=96, kernel_size=[3, 3], strides=strides, use_bias=False,input_shape=in_shape),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
          keras.layers.Conv2D(filters=96, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
          keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False),
          keras.layers.Conv2D(filters=64, kernel_size=[1, 1], strides=[1, 1], use_bias=False),
          keras.layers.BatchNormalization(momentum=0.1),
          keras.layers.ReLU(),
          keras.layers.GlobalAveragePooling2D(),
          keras.layers.Dense(units=8),
      ])
    NUM_EPOCHS=30

print(model.summary())
print("MODEL GENERATED")

cp_model="./cp_std"
cp_callback = keras.callbacks.ModelCheckpoint(
    filepath=cp_model, #OVERWRITE->SAME FOLDER
    monitor='val_sparse_categorical_accuracy',
    save_best_only=True,
    save_weights_only=True,
    mode='auto',
    save_freq='epoch',
)

print("TRAIN")
model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
history = model.fit(train_ds, verbose=2, epochs=NUM_EPOCHS, validation_data = (val_ds), callbacks=[cp_callback])
model.load_weights(cp_model)

print("EVALUATION")
loss,metric=model.evaluate(test_ds)

if args.version=="a":
    saved_model_dir="./model_standard_ex2_a"
if args.version=="b":
    saved_model_dir="./model_standard_ex2_b"
if args.version=="c":
    saved_model_dir="./model_standard_ex2_c"
run_model = tf.function(lambda x: model(x))
if mfcc:
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 49, 10, 1],tf.float32))
else:
    concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, size_in, size_in, 1],tf.float32))
model.save(saved_model_dir, signatures=concrete_func)

'''
print("PRUNING")
pruning_params = {'pruning_schedule':
                    tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.3,
                    final_sparsity=0.9,
                    begin_step=len(train_ds)*5,
                    end_step=len(train_ds)*15
                    #begin_step=0,
                    #end_step=len(train_ds)*20
                    )
}
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_pruned = prune_low_magnitude(model, **pruning_params)

es_callback = keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_accuracy', patience=1, verbose=1)
callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]

model_pruned.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model_pruned.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)

saved_model_dir_pruned="./model_pruned_ex2"
run_model_pruned = tf.function(lambda x: model_pruned(x))
if mfcc:
    concrete_func = run_model_pruned.get_concrete_function(tf.TensorSpec([1, 49, 10, 1],tf.float32))
else:
    concrete_func = run_model_pruned.get_concrete_function(tf.TensorSpec([1, size_in, size_in, 1],tf.float32))
model_pruned.save(saved_model_dir_pruned, signatures=concrete_func)
'''

print("TFLITE")
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
if args.version=="a":
    tflite_model_dir_zipped="./Group20_kws_a.tflite.zlib"
if args.version=="b":
    tflite_model_dir_zipped="./Group20_kws_b.tflite.zlib"
if args.version=="c":
    tflite_model_dir_zipped="./Group20_kws_c.tflite.zlib"
with open(tflite_model_dir_zipped, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)
    
if args.version=="a":
    tflite_model_dir="./Group20_kws_a.tflite"
if args.version=="b":
    tflite_model_dir="./Group20_kws_b.tflite"
if args.version=="c":
    tflite_model_dir="./Group20_kws_c.tflite"
with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)

normal_model=os.path.getsize("./model_standard_ex2/saved_model.pb")
lite_model_zipped=os.path.getsize(tflite_model_dir_zipped)
lite_model=os.path.getsize(tflite_model_dir)
print(f"Normal Model: {normal_model}, Lite Model: {lite_model}, Lite Model Zipped: {lite_model_zipped}")

#INTERPRETER
interpreter = tflite.Interpreter(model_path=tflite_model_dir)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details() #DICTIONARY
output_details = interpreter.get_output_details ()

test_ds_saved = test_ds.unbatch().batch(1)

inputs=[]
outputs=[]
for el in test_ds_saved:
    my_input=el[0].numpy()
    interpreter.set_tensor(input_details[0]['index'], my_input) #COPY DATA INTO THE BUFFER
    interpreter.invoke() #CALL THE INTERPRETER
    my_output = interpreter.get_tensor(output_details[0]['index']) #SAVE THE OUTPUT
    inputs.extend(el[1].numpy())
    outputs.extend(my_output)

correct=0
for el1,el2 in zip(inputs,outputs):
  el2=np.array(el2)
  if np.argmax(el2)==el1:
    correct+=1
print(correct/len(inputs)*100)
