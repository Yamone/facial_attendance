# coding:utf-8
##
 #  (c) Copyright by nexidea.
##
from sklearn.neighbors import KNeighborsClassifier
import tensorflow as tf
import argparse
from utils.read_data import read_image,load_image_file
import time
import numpy as np
import cv2
import pickle
import os
from pathlib import Path
import shutil


def load_model(model):
    # Check if the model is a model directory (containing a metagraph and a checkpoint file)
    #  or if it is a protobuf file with a frozen graph
    model_exp = os.path.expanduser(model)
    if (os.path.isfile(model_exp)):
        print('Model filename: %s' % model_exp)
        with tf.gfile.FastGFile(model_exp, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name='')

    else:
        print('Model directory: %s' % model_exp)
        meta_file, ckpt_file = get_model_filenames(model_exp)

        print('Metagraph file: %s' % meta_file)
        print('Checkpoint file: %s' % ckpt_file)

        saver = tf.train.import_meta_graph(os.path.join(model_exp, meta_file))
        saver.restore(tf.get_default_session(), os.path.join(model_exp, ckpt_file))


def get_model_filenames(model_dir):
    files = os.listdir(model_dir)
    meta_files = [s for s in files if s.endswith('.meta')]
    if len(meta_files) == 0:
        raise ValueError('No meta file found in the model directory (%s)' % model_dir)
    elif len(meta_files) > 1:
        raise ValueError('There should not be more than one meta file in the model directory (%s)' % model_dir)
    meta_file = meta_files[0]
    ckpt = tf.train.get_checkpoint_state(model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        ckpt_file = os.path.basename(ckpt.model_checkpoint_path)
        return meta_file, ckpt_file

    meta_files = [s for s in files if '.ckpt' in s]
    max_step = -1
    for f in files:
        step_str = re.match(r'(^model-[\w\- ]+.ckpt-(\d+))', f)
        if step_str is not None and len(step_str.groups()) >= 2:
            step = int(step_str.groups()[1])
            if step > max_step:
                max_step = step
                ckpt_file = step_str.groups()[0]
    return meta_file, ckpt_file

def train_save_classifier(embeddings_array,labels,class_name_array,classifier_output_path):
    knn_model = KNeighborsClassifier(n_neighbors=5, algorithm = 'auto',weights = 'distance')
    knn_model.fit(embeddings_array, labels)
    print("Finish training!")
    with open(classifier_output_path,'wb') as f:
        pickle.dump((knn_model,class_name_array),f)

def parse_function(filename,label):

    image_string = tf.read_file(filename)
    img = tf.image.decode_jpeg(image_string,channels=3)
    img = tf.image.resize_images(img,[112,112])
    img = tf.image.convert_image_dtype(img,tf.float32)
    img = tf.subtract(img, 127.5)
    img = tf.multiply(img,  0.0078125)
    # img = tf.image.adjust_gamma(img,gamma=1,gain=1)
    # img = tf.image.random_flip_left_right(img)
    label = tf.cast(label,dtype = tf.int64)
    return img, label

def get_parser():
    parser = argparse.ArgumentParser(description='parameters to train net')
    #new
    parser.add_argument('--model_dir', type=str, default = '../models/MobileFaceNet/MobileFaceNet.pb',
                        help='Norm to use for prelogits norm loss.')
    parser.add_argument('--data_dir', type=str, default = '../final_processing_data',
                        help='Norm to use for prelogits norm loss.')
    parser.add_argument('--train_batch_size', default=60, help='batch size to train network')
    args = parser.parse_args()
    return args


def train():
    print("Main function.")
    classifier_path = "../models/classifier_output"

    # classifier file path
    if not os.path.exists(classifier_path):
        os.makedirs(classifier_path)

    classifier_output_path = os.path.join(classifier_path,"classify_result.pkl")

    with tf.Graph().as_default():
        args = get_parser()
        embeddings_array = []
        labels = []
        class_name_array = []

        #load model
        load_model(args.model_dir)

        input_node = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        #embeddings_size = embeddings.get_shape()[1]

        # prepare train dataset
        img,label = load_image_file(args.data_dir) # "0" index for train data
        dataset = tf.data.Dataset.from_tensor_slices((img,label))
        dataset = dataset.map(parse_function,num_parallel_calls=4)
        dataset = dataset.batch(args.train_batch_size)
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()

        with tf.Session() as sess:
            sess.run(iterator.initializer)
            while True:
                try:
                    images_train, labels_train = sess.run(next_element)
                    feed_dict = {input_node:images_train}
                    emb = sess.run(embeddings,feed_dict = feed_dict)
                    labels.extend(labels_train)
                    embeddings_array.extend(emb)

                except tf.errors.OutOfRangeError:
                    print("Creating embeddings!")
                    # sess.run(iterator.initializer)
                    break

            #convert numpy array
            embeddings_array = np.asarray(embeddings_array)
            labels = np.asarray(labels)
            print(embeddings_array.shape, labels.shape)

            #read class name file
            f = open("output_labels.txt","r")
            class_name_array.extend(f.read().splitlines())
            f.close()

            print("Start training!!!")
            train_save_classifier(embeddings_array,labels,class_name_array,classifier_output_path)

def main():
    if os.path.exists("../models/classifier_output/classify_result.pkl"):


        if os.path.exists("../models/deleted_model/classify_result.pkl"):
            os.remove("../models/deleted_model/classify_result.pkl")
            # time.sleep(5)

        shutil.move("../models/classifier_output/classify_result.pkl", "../models/deleted_model/classify_result.pkl")
        print("Exist")
    else:
        print("Not exist")

    # os.rename("../models/classifier_output/classify_result.pkl","../models/deleted_model")
    # time.sleep(5)
    # print("Train model")
    train()

if __name__ == '__main__':
    main()
