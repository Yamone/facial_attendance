# coding:utf-8
##
 #  (c) Copyright by nexidea.
##

import tensorflow as tf
import os
import sys
import pickle
import numpy as np
import cv2
from preprocess.align_dlib import AlignDlib
import time

#new
crop_dim = 112
model_dir = '../models/MobileFaceNet/MobileFaceNet.pb'
classifier_dir = '../models/classifier_output/classify_result.pkl'

align_dlib = AlignDlib(os.path.join(os.path.dirname(__file__), './preprocess/shape_predictor_68_face_landmarks.dat'))#new

#new function
def predict(features, knn_clf=None, model_path=None, distance_threshold=0.4):

    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf,label = pickle.load(f)

    # If no faces are found in the image, return an empty result.
    if len(features) == 0:
        return []

    # Use the KNN model to find the best matches for the test face
    #return shortest distance value and its index
    closest_distances = knn_clf.kneighbors(features, n_neighbors=3)

    # print("closet_distance",closest_distances) #(array([[0.46375529]]), array([[3566]]))
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(features))]

    #Return probability estimates for the test data X.
    y_scores = knn_clf.predict_proba(features)

    # Predict classes and remove classifications that aren't within the threshold
    return [(label[pred]) if rec else ("unknown") for pred, rec in zip(knn_clf.predict(features), are_matches)],y_scores

#new function
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
        print("No file found")
        sys.exit()

def adjust_gamma(image, gamma=1.5):
    # print("Gamma adjust!")
    # build a lookup table mapping the pixel values [0, 255] to
    # their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                      for i in np.arange(0, 256)]).astype("uint8")

    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def identify_me(image):
     #load train model
    load_model(model_dir)

    input_node = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    embedding_size = embeddings.get_shape()[1]
    print("Type is",type(embedding_size), embedding_size)
    emb_array = np.zeros((1, embedding_size))


    sess = tf.Session()

    #preprocess image
    #read an image
    # img1 = cv2.imread(input_dir)
    img1=cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    #img1=cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)

    #print("image",image)

    #convert BGR to RGB format
    image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    #detect faces in an image
    face = align_dlib.getLargestFaceBoundingBox(image)
    # print("Number of face detected : ",detected_face)
    print("Names of people in image are : \n")
    aligned = align_dlib.align(crop_dim, image, face, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

    if aligned is not None:

        #convert cv2 default format(BGR) to RGB(model acceptance format)
        aligned_image = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        #gamma adjust image
        adjusted_image = adjust_gamma(aligned_image)

        #convert cv2 default format(BGR) to RGB(model acceptance format)
        aligned_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

        aligned_image = aligned_image.astype('float32')
        aligned_image = aligned_image - 127.5
        aligned_image = aligned_image * 0.0078125


        #expand dimension from (112,112,3) to (1,112,112,3)
        aligned_image = np.expand_dims(aligned_image, axis=0) #(1,112,112,3)

        # print("Updated Aligned face shape",aligned_image.shape)

        feed_dict = {input_node:aligned_image}
        emb_array[0,:] = sess.run(embeddings,feed_dict = feed_dict)

        user_id,prob = predict(features=emb_array,model_path=classifier_dir)

        print(user_id,"\n")

        result = user_id[0]
        start = time.time()
        # cv2.imwrite('%s/%s_userId=%s.png' % ('static/img/logs',start, result), img1)

    else:
        result = "unknown"
        print("No face detected.")
    sess.close()
    return result

def identify_me_with_frame(image):
     #load train model
    load_model(model_dir)

    input_node = tf.get_default_graph().get_tensor_by_name("img_inputs:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    embedding_size = embeddings.get_shape()[1]
    print("Type is",type(embedding_size), embedding_size)
    emb_array = np.zeros((1, embedding_size))


    sess = tf.Session()

    #preprocess image
    #read an image
    # img1 = cv2.imread(input_dir)
    #img1=cv2.imdecode(np.fromstring(image.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img1=cv2.imdecode(np.fromstring(image, np.uint8), cv2.IMREAD_UNCHANGED)

    #print("image",image)

    #convert BGR to RGB format
    image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)

    #detect faces in an image
    face = align_dlib.getLargestFaceBoundingBox(image)
    # print("Number of face detected : ",detected_face)
    print("Names of people in image are : \n")
    aligned = align_dlib.align(crop_dim, image, face, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)

    if aligned is not None:

        #convert cv2 default format(BGR) to RGB(model acceptance format)
        aligned_image = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)

        #gamma adjust image
        adjusted_image = adjust_gamma(aligned_image)

        #convert cv2 default format(BGR) to RGB(model acceptance format)
        aligned_image = cv2.cvtColor(adjusted_image, cv2.COLOR_BGR2RGB)

        aligned_image = aligned_image.astype('float32')
        aligned_image = aligned_image - 127.5
        aligned_image = aligned_image * 0.0078125


        #expand dimension from (112,112,3) to (1,112,112,3)
        aligned_image = np.expand_dims(aligned_image, axis=0) #(1,112,112,3)

        # print("Updated Aligned face shape",aligned_image.shape)

        feed_dict = {input_node:aligned_image}
        emb_array[0,:] = sess.run(embeddings,feed_dict = feed_dict)

        user_id,prob = predict(features=emb_array,model_path=classifier_dir)

        print(user_id,"\n")

        result = user_id[0]
        start = time.time()
        # cv2.imwrite('%s/%s_userId=%s.png' % ('static/img/logs',start, result), img1)

    else:
        result = "unknown"
        print("No face detected.")
    sess.close()
    return result
