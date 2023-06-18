import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

from knn import *
from inference import *

import cv2
import time
from datetime import timedelta
import random
import warnings
import tensorflow as tf
import numpy as np
import pandas
from configs import *
import pyarrow.parquet as pq
import pyarrow as pa
import csv

warnings.filterwarnings("ignore")
random.seed(42)

def display_confusion_matrix(y_true, y_pred):
    print(y_pred)
    print(y_true)
    labels =np.unique(np.array(y_true))
    cm = confusion_matrix(y_true, y_pred, normalize='true', labels=labels)
    df_cm = pandas.DataFrame(cm).transpose()

    df_cm.to_csv(path_as_string(results_path)+"/confusion_matrix.csv")
    df_cm.to_html(path_as_string(results_path)+"/confusion_matrix.html")

    cr = classification_report(y_true, y_pred, output_dict=True)
    df = pandas.DataFrame(cr).transpose()

    df.to_csv(path_as_string(results_path)+"/classification_report.csv")
    df.to_html(path_as_string(results_path)+"/classification_report.html")

def load_model():
    model = tf.keras.models.load_model(path_as_string(model_path) + '/' + model_name)
    model_without_output = tf.keras.models.Model(inputs=model.input, outputs=model.layers[-1].input)
    print(model_without_output.summary())
    return model_without_output

def get_images_paths(image_folders):
    out_list_ds = []
    
    for image_folder in image_folders:
        list_ds = list_files(image_folder)
        for element_ds in list_ds:
            out_list_ds.append(element_ds.numpy().decode('utf-8'))
    return out_list_ds

def get_embedings_paths():
    print("split_dataset_embedings_path", path_as_string(split_dataset_embedings_path))
    image_folders = [path_as_string(split_dataset_embedings_path)]
    out_list_ds = get_images_paths(image_folders)
    
    return out_list_ds

def get_image_dataset_from_directory(folder_path):
    return tf.keras.utils.image_dataset_from_directory(path_as_string(folder_path),
        image_size=(img_input_size), 
        batch_size=batch_size,
        interpolation=image_interpolation
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

def get_image_from_directory(image_path):
    picture = tf.keras.preprocessing.image.load_img(path=image_path, 
        grayscale=False, color_mode='rgb', 
        target_size=img_input_size, interpolation=image_interpolation
    )
    return picture

def get_test_images():
    test_images_paths = get_images_paths([path_as_string(split_dataset_test_path)])
    random.shuffle(test_images_paths)
    
    return test_images_paths

def create_image_feature(image_path, prediction):
    # print(image_path)
    parts = image_path.split("/")
    image_name = parts[-1].split(".")[0]
    label = parts[-2]
    # print(label)

    return {"image_path": image_path, 
            "image_name": image_name, 
            "class": label,
            "embedding":np.array(prediction[0])}

def create_image_datarecord(model, image_path):
    picture = get_image_from_directory(image_path)
    picture = tf.keras.preprocessing.image.img_to_array(picture)
    picture = np.array([picture])  # Convert single image to a batch.
    prediction = model.predict(picture, batch_size = batch_size)
    
    return create_image_feature(image_path, prediction)

def create_images_dataset(model, images_paths):
    elements = []
    for image_path in images_paths:
        print(image_path)
        elements.append(create_image_datarecord(model, image_path))

    elements.sort(key=lambda embeding: embeding.get('image_name'))

    return elements

def test_model():
    print("load_model")
    model = load_model()
    print("img_input_size:" + str(IMG_SIZE))

    images_paths = get_embedings_paths()
    image_dataset = create_images_dataset(model, images_paths)
    print("image_dataset: "+str(len(image_dataset)))

    test_paths = get_test_images()
    print(len(test_paths))
    acc = 0
    n_acc = 0
    y_true = []
    y_pred = []
    process_time = []

    embeddings = np.stack([ element["embedding"] for element in image_dataset ])
    index = build_index(embeddings)

    fieldnames = ['query', 'path',
                'first', 'path_first', 'distance_1', 
                'second', 'path_second', 'distance_2', 
                'third', 'path_third', 'distance_3', 
                'fourth', 'path_fourth', 'distance_4', 
                'time']
    rows = []
    for test_path in test_paths:
        print(test_path)
        real_time = 0

        # Load query and emb
        start = time.time()
        test_image = create_image_datarecord(model, test_path)

        # Build index
        embedding = test_image["embedding"]

        results = search_results_2(index, embedding, image_dataset)
        end = time.time()
        real_time = (end - start)
        process_time.append(real_time)
        # Display results
        display_results(test_image, results)
        
        # Calculate accuracy
        top_knn = results[0]
        element = top_knn['element']
        predicted_class = element['class']
        real_class = test_image['class']
        print('query', real_class, 'path', test_path)
        print('first', results[0]['element']['class'], 'path_first', results[0]['element']['image_path'], 'distance_1', results[0]['distance'])

        y_true.append(real_class)
        y_pred.append(predicted_class)

        row = {
            'query': real_class, 'path': test_path,
            'first': results[0]['element']['class'], 'path_first': results[0]['element']['image_path'], 'distance_1': results[0]['distance'], 
            'second': results[1]['element']['class'], 'path_second': results[1]['element']['image_path'], 'distance_2': results[1]['distance'], 
            'third': results[2]['element']['class'], 'path_third': results[2]['element']['image_path'], 'distance_3': results[2]['distance'], 
            'fourth': results[3]['element']['class'], 'path_fourth': results[3]['element']['image_path'], 'distance_4': results[3]['distance'], 
            'time':real_time, 
            }
        rows.append(row)

        n_acc+=1
        if real_class == predicted_class:
            acc += 1
        print(f"processing: {test_path}, mean time: {real_time}, mean acc: {acc / n_acc}")


    print(f"\n Max time in second: {np.max(process_time)}")
    print(f"\n Min time in second: {np.min(process_time)}")
    print(f"\n mean time in second: {np.mean(process_time)}")
    print(f"\n std time in second: {np.std(process_time)}")

    with open(path_as_string(results_path) + '/'+ 'results.csv', 'w', 
            encoding='UTF8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

        writer_summary = csv.DictWriter(f, fieldnames=['type', 'val'])
        writer_summary.writeheader()
        writer_summary.writerows([
            {'type': 'Max time in second:', 'val' :np.max(process_time)},
            {'type': 'Min time in second:', 'val' :np.min(process_time)},
            {'type': 'Mean time in second:', 'val' :np.mean(process_time)},
            {'type': 'Std time in second:', 'val' :np.std(process_time)},
        ])

    with open(path_as_string(log_path) +'/emb_fast01.txt', 'a') as file:
        file.write(str("Mean accuracy: " + str(acc / n_acc) + ", mean time: " + str(np.mean(process_time)) + ", std time: " + str(np.std(process_time)) + '\n'))

    display_confusion_matrix(y_true, y_pred)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
for gpu in physical_devices:
            tf.config.experimental.set_memory_growth(gpu, True)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        tf.config.set_visible_devices(gpus[0], 'GPU')

        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
        
        test_model()
    except RuntimeError as e:
        print(e)
else:
    print("bez gpu")
    test_model()