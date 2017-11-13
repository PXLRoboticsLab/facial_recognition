#!/usr/bin/env python
import argparse
import rospy
import math
import cv2
import pickle
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc
import align.detect_face
import tensorflow as tf
import facenet
import time
import glob
import schedule
import os
from facial_recognition.msg import personData
from facial_recognition.msg import personDataArray
from std_msgs.msg import String
from rostopic import ROSTopicHz

changed = []


def get_hz(topic, window_size=-1, filter_expr=None):
    rates = []
    rt = ROSTopicHz(window_size, filter_expr=filter_expr)
    sub = rospy.Subscriber(topic, Image, rt.callback_hz)
    while not rospy.is_shutdown():
        if len(rt.times) > 0:
            n = len(rt.times)
            mean = sum(rt.times) / n
            rate = 1. / mean if mean > 0. else 0
            rates.append(rate)
            if len(rates) >= 3:
                sub.unregister()
                return np.average(rates)
        rospy.sleep(1)


def change_oldest_picture(folder, person, new_picture):
    if person != "Unknown" and person not in changed:
        changed.append(person)
        full_path = os.path.join(folder, person.replace(" ", "_"))
        list_of_files = glob.glob(full_path + "/*.png")
        oldest_file = min(list_of_files, key=os.path.getctime)
        misc.imsave(oldest_file, new_picture)
        print(oldest_file)


def reset_list():
    global changed
    changed = []


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(gpu_memory_fraction / 2))
        #gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, None)
    return pnet, rnet, onet


def align_data(image_list, image_size, margin, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor

    img_list = []
    boxes = []

    for x in xrange(len(image_list)):
        img_size = np.asarray(image_list[x].shape)[0:2]
        bounding_boxes, _ = align.detect_face.detect_face(image_list[x], minsize, pnet, rnet, onet, threshold, factor)
        nrof_samples = len(bounding_boxes)
        if nrof_samples > 0:
            for i in xrange(nrof_samples):
                if bounding_boxes[i][4] > 0.95:
                    det = np.squeeze(bounding_boxes[i, 0:4])
                    bb = np.zeros(4, dtype=np.int32)
                    bb[0] = np.maximum(det[0] - margin / 2, 0)
                    bb[1] = np.maximum(det[1] - margin / 2, 0)
                    bb[2] = np.minimum(det[2] + margin / 2, img_size[1])
                    bb[3] = np.minimum(det[3] + margin / 2, img_size[0])
                    cropped = image_list[x][bb[1]:bb[3], bb[0]:bb[2], :]
                    # Check if found face isn't to blurry
                    if cv2.Laplacian(cropped, cv2.CV_64F).var() > 100:
                        aligned = misc.imresize(cropped, (image_size, image_size), interp='bilinear')
                        prewhitened = facenet.prewhiten(aligned)
                        img_list.append(prewhitened)
                        boxes.append(bounding_boxes[i])

    if len(img_list) > 0:
        images = np.stack(img_list)
        return images, boxes

    return None, None


class PersonClassifier():
    BATCH_SIZE = 90
    IMAGE_SIZE = 160
    MARGIN = 44

    _pnet = None
    _rnet = None
    _onet = None
    _model = None
    _classifier = None
    _data_dir = None
    _camera_topic = None
    _id = None
    _counter = 0
    _model_pkl = None
    _class_names = None

    def __init__(self, model, classifier_filename, data_dir, camera_topic, id, memory, pnet, rnet, onet):
        self._model = model
        self._classifier = classifier_filename
        self._data_dir = data_dir
        self._camera_topic = camera_topic
        self._id = id
        self._pnet = pnet
        self._rnet = rnet
        self._onet = onet

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(memory/2))

        self._hz = int((get_hz(self._camera_topic) / 5) - 0.5)

        # Create the topic where the data will be published
        self.pub = rospy.Publisher("focus_vision/person/identification", personDataArray, queue_size=30)
        self.pub_image = rospy.Publisher("focus_vision/image/identification/" + self._id, Image, queue_size=30)
        self.pub_unknown_person = rospy.Publisher("focus_vision/image/unidentified", Image, queue_size=30)

        with tf.gfile.FastGFile(self._model, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

        with tf.Session(config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False)) as self.sess:
            # Retrieve the required input and output tensors for classification from the model
            self.images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
            self.embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
            self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
            self.embedding_size = self.embeddings.get_shape()[1]

            self.bridge = CvBridge()

            print('Loaded classifier model from file "%s"' % self._classifier)
            with open(self._classifier, 'rb') as infile:
                (self._model_pkl, self._class_names) = pickle.load(infile)

            # Subscribe to the camera topic provided in the args
            self.image_sub = rospy.Subscriber(self._camera_topic, Image, self.classify)

            # Subscribe to the topic providing the latest SVM classifier
            self.model_sub = rospy.Subscriber("/trained_model", String, self.reload_model)

    def reload_model(self, data):
        self._classifier = data.data
        with open(self._classifier, 'rb') as infile:
            (self._model_pkl, self._class_names) = pickle.load(infile)

    def classify(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        if self._counter == self._hz:
            self._counter = 0

            # Resize the image to a standard width and height
            cv_image = cv2.resize(cv_image, (1280, 720))
            # Change the color format of the image from BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Align the faces in the list properly before classification
            images, boxes = align_data([cv_image], self.IMAGE_SIZE, self.MARGIN, self._pnet, self._rnet, self._onet)
            # Check if after aligning the list was returned and is not None
            if images is not None:
                # Prepare for classification of images
                nrof_images = len(images)
                nrof_batches_per_epoch = int(math.ceil(1.0 * nrof_images / self.BATCH_SIZE))
                emb_array = np.zeros((nrof_images, self.embedding_size))
                for i in range(nrof_batches_per_epoch):
                    start_index = i * self.BATCH_SIZE
                    end_index = min((i + 1) * self.BATCH_SIZE, nrof_images)
                    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                    emb_array[start_index:end_index, :] = self.sess.run(self.embeddings, feed_dict=feed_dict)

                predictions = self._model_pkl.predict_proba(emb_array)
                best_class_indices = np.argmax(predictions, axis=1)
                best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]

                msgs = personDataArray()

                for i in range(len(best_class_indices)):
                    print(
                        '%4d  %s: %.3f' % (
                            i, self._class_names[best_class_indices[i]], best_class_probabilities[i]))
                    msg = personData(self._class_names[best_class_indices[i]], self._id,
                                     best_class_probabilities[i])
                    if best_class_probabilities[i] > 0.6:
                        cv_image = cv2.rectangle(cv_image, (int(boxes[i][0]), int(boxes[i][1])),
                                                 (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                        cv_image = cv2.putText(cv_image, self._class_names[best_class_indices[i]],
                                               (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                               (0, 255, 0), 1, cv2.LINE_AA)
                        if best_class_probabilities[i] > 0.8:
                            change_oldest_picture(self._data_dir, self._class_names[best_class_indices[i]], images[i])
                    if self._class_names[best_class_indices[i]] == "Unknown":
                        misc.imsave('/home/maarten/Pictures/tmp.png', images[i])
                        img = cv2.imread('/home/maarten/Pictures/tmp.png')
                        self.pub_unknown_person.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

                    msgs.data.append(msg)
                self.pub.publish(msgs)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
        else:
            self._counter += 1


if __name__ == "__main__":
    rospy.init_node("person_classification")

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file')
    parser.add_argument('classifier',
                        help='Path to the classifier model file name as a pickle (.pkl) file.')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing classifier data.')
    parser.add_argument('camera_topic', type=str,
                        help='The ros topic name from the camera stream you want to analyze.')
    parser.add_argument('--id', type=str, help='The id of the camera being analized.', default='dlink_1')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.40)
    # parser.add_argument('--location', type=float, nargs='2',
    # help='The X and Y location of the camera, use this option only for stationary cameras.')

    args = parser.parse_args()

    schedule.every().day.at("11:59").do(reset_list)
    schedule.every().day.at("23:59").do(reset_list)

    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    classifier = PersonClassifier(args.model, args.classifier, args.data_dir, args.camera_topic, args.id,
                                  args.gpu_memory_fraction, pnet, rnet, onet)
    while True:
        schedule.run_pending()
        time.sleep(5)
