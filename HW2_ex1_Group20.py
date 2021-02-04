import argparse
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import tensorflow.lite as tflite
import tensorflow_model_optimization as tfmot
import zlib


parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True, help='version name')
args = parser.parse_args()

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

#3.1
zip_path = tf.keras.utils.get_file(
   origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    )
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n*0.7)]
val_data = data[int(n*0.7):int(n*0.9)]
test_data = data[int(n*0.9):]
print("DATA DOWNLOADED")
#print(type(train_data))
#print(train_data.shape)
#print(type(train_data[0][0]))
mean = train_data.mean(axis=0)
std = train_data.std(axis=0)

input_width = 6
LABEL_OPTIONS = 2

class WindowGenerator:
    def __init__(self, input_width, label_options, mean, std):
        self.input_width = input_width
        self.label_options = label_options
        self.mean = tf.reshape(tf.convert_to_tensor(mean), [1, 1, 2])
        self.std = tf.reshape(tf.convert_to_tensor(std), [1, 1, 2])

    def split_window(self, features):
        #print(features.shape)
        input_indeces = np.arange(self.input_width)
        inputs = features[:, :6, :]
        #print(inputs)

        if self.label_options < 2:
            labels = features[:, -1, self.label_options]
            labels = tf.expand_dims(labels, -1)
            num_labels = 1
        else:
            labels = features[:, 6:, :]
            num_labels = 2

        inputs.set_shape([None, self.input_width, 2])
        labels.set_shape([None, self.input_width, num_labels])

        return inputs, labels

    def normalize(self, features):
        features = (features - self.mean) / (self.std + 1.e-6)

        return features

    def preprocess(self, features):
        inputs, labels = self.split_window(features)
        inputs = self.normalize(inputs)

        return inputs, labels

    def make_dataset(self, data, train):
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
                data=data,
                targets=None,
                sequence_length=self.input_width+6,
                sequence_stride=1,
                batch_size=32)
        ds = ds.map(self.preprocess) #32,6,2
        ds = ds.cache()
        if train is True:
            ds = ds.shuffle(100, reshuffle_each_iteration=True)

        return ds

generator = WindowGenerator(input_width, LABEL_OPTIONS, mean, std)
train_ds = generator.make_dataset(train_data, True)
val_ds = generator.make_dataset(val_data, False)
test_ds = generator.make_dataset(test_data, False)
print("DATASET GENERATED")

class MultiOutputMAE(tf.keras.metrics.Metric): 
    def __init__(self, name='mean_abs_error', **kwargs):
        super().__init__(name, **kwargs)
        self.total=self.add_weight('total', initializer='zeros', shape=[2]) #abbiamo 2 output
        self.count =self.add_weight('count', initializer='zeros')

    #metodi chiamati quando ho fit ed evaluate

    def reset_states(self):
        self.count.assign(tf.zeros_like(self.count))
        self.total.assign(tf.zeros_like(self.total))
        return

    def update_state(self, y_true, y_pred, sample_weight=None):
        error= tf.abs(y_pred-y_true)
        error=tf.reduce_mean(error, axis=(1,0))
        #error=tf.reduce_mean(error, axis=0)
        #mae calcolato su un singolo batch
        self.total.assign_add(error)
        self.count.assign_add(1)
        return
    #chiamato alla fine dell'iterazione
    def result(self): 
        result = tf.math.divide_no_nan(self.total, self.count)
        #print(type(result))
        return result
    
units=1
if LABEL_OPTIONS==2:
    units=12

if args.version == "b":
    model = keras.Sequential([
        keras.layers.Flatten(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(units),
        keras.layers.Reshape((6,2)),
    ])
else:
    model = keras.Sequential([
        keras.layers.Conv1D(16, 3, activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(units),
        keras.layers.Reshape((6,2)),
    ])
print("MODEL GENERATED")

def my_schedule(epoch, lr):
    if epoch < 12:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
lr_callback = keras.callbacks.LearningRateScheduler(
    my_schedule,
    verbose=0,
)

met=MultiOutputMAE()
opt=tf.keras.optimizers.Adam(
    learning_rate=0.001, #0.01,0.0001
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name="Adam",
)

print("TRAIN")
model.compile(optimizer=opt,loss=tf.keras.losses.MeanSquaredError(),metrics=[met])
history = model.fit(train_ds, verbose=1, epochs=20, validation_data = (val_ds), callbacks=[lr_callback])

print("EVALUATION")
loss,metric = model.evaluate(test_ds)
print(metric)

if args.version=="a":
    saved_model_dir="./model_standard_ex1_a"
if args.version=="b":
    saved_model_dir="./model_standard_ex1_b"
run_model = tf.function(lambda x: model(x))
concrete_func = run_model.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
model.save(saved_model_dir, signatures=concrete_func)
#model_saved=tf.keras.models.load_model("/content/model_standard/",custom_objects={'MultiOutputMAE':MultiOutputMAE()})

print("PRUNING")
pruning_params = {'pruning_schedule':
                    tfmot.sparsity.keras.PolynomialDecay(
                    initial_sparsity=0.30,
                    final_sparsity=0.9,
                    begin_step=len(train_ds)*5,
                    end_step=len(train_ds)*15)
}

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
model_pruned = prune_low_magnitude(model, **pruning_params)
es=keras.callbacks.EarlyStopping(monitor='val_loss', patience=2, verbose=1)
callbacks = [tfmot.sparsity.keras.UpdatePruningStep(), lr_callback]
input_shape = [32, 6, 2]
model_pruned.build(input_shape)
model_pruned.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=[met])
model_pruned.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
model_pruned = tfmot.sparsity.keras.strip_pruning(model_pruned)

if args.version=="a":
    saved_model_dir_pruned="./model_pruned_ex1_a"
if args.version=="b":
    saved_model_dir_pruned="./model_pruned_ex1_b"
run_model_pruned = tf.function(lambda x: model_pruned(x))
concrete_func = run_model_pruned.get_concrete_function(tf.TensorSpec([1, 6, 2],tf.float32))
model_pruned.save(saved_model_dir_pruned, signatures=concrete_func)

''' WEIGHTS CLUSTERING
def apply_wc_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.clustering.keras.cluster_weights(layer, 
                                                      number_of_clusters=8, #ALSO TRIED WITH 4 and 16
                                                      cluster_centroids_init=tfmot.clustering.keras.CentroidInitialization.LINEAR)
    return layer
    
model_for_clust = tf.keras.models.clone_model(
    model_pruned,
    clone_function=apply_wc_to_dense,
)
callbacks = [lr_callback]
model_for_clust.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=[met])
model_for_clust.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
model_clust = tfmot.clustering.keras.strip_clustering(model_for_clust)
'''

''' QUANTIZATION AWARE
def apply_qa_to_dense(layer):
    if isinstance(layer, tf.keras.layers.Dense):
        return tfmot.quantization.keras.quantize_model(layer)
    return layer
    
model_qa = tf.keras.models.clone_model(
    model_clust,
    clone_function=apply_wc_to_dense,
)
model_qa.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError(),metrics=[met])
model_qa.fit(train_ds, epochs=20, validation_data=val_ds, callbacks=callbacks)
'''

print("TFLITE")
''' WEIGHTS+ACTIVATION
def representative_dataset_gen():
    for x, _ in train_ds.take(1000):
        yield [x]
'''
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir_pruned)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
''' WEIGHTS+ACTIVATION
#converter.representative_dataset = representative_dataset_gen
#converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#converter.inference_input_type = tf.uint8
#converter.inference_output_type = tf.uint8
'''
tflite_model = converter.convert()

if args.version=="a":
    tflite_model_dir_zipped="./Group20_th_a.tflite.zlib"
if args.version=="b":
    tflite_model_dir_zipped="./Group20_th_b.tflite.zlib"
with open(tflite_model_dir_zipped, 'wb') as fp:
    tflite_compressed = zlib.compress(tflite_model)
    fp.write(tflite_compressed)
    
if args.version=="a":
    tflite_model_dir="./Group20_th_a.tflite"
if args.version=="b":
    tflite_model_dir="./Group20_th_b.tflite"
with open(tflite_model_dir, 'wb') as fp:
    fp.write(tflite_model)
normal_model=os.path.getsize("./model_standard_ex1/saved_model.pb")

lite_model_zipped=os.path.getsize(tflite_model_dir_zipped)
lite_model=os.path.getsize(tflite_model_dir)
print(f"Normal Model: {normal_model}, Lite Model: {lite_model}, Lite Model Zipped: {lite_model_zipped}")

interpreter = tflite.Interpreter(model_path=tflite_model_dir)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details() #DICTIONARY
output_details = interpreter.get_output_details ()

print("TFLITE EVALUATION")
test_ds_saved = test_ds.unbatch().batch(1)
inputs=[]
outputs=[]
for el in test_ds_saved:
    input=el[0].numpy()
    interpreter.set_tensor(input_details[0]['index'], input) #COPY DATA INTO THE BUFFER
    interpreter.invoke() #CALL THE INTERPRETER
    my_output = interpreter.get_tensor(output_details[0]['index']) #SAVE THE OUTPUT
    inputs.extend(el[1].numpy())
    outputs.extend(my_output)
inputs=np.array(inputs)
outputs=np.array(outputs)
err=np.abs(inputs-outputs)
#print(err.shape)
tmp=np.mean(err, axis=1)
#print(tmp.shape)
mae=np.mean(tmp, axis=0)
print(f'FINAL RESULT:{mae}')





