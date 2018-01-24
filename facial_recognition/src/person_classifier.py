#!/usr/bin/env python
import argparse
import rospy
import cv2
import pickle
import facenet
import time
import datetime
import threading
import itertools
import glob
import os
import align.detect_face
import numpy as np
import tensorflow as tf
from facial_recognition.msg import personData
from facial_recognition.msg import personDataArray
from std_msgs.msg import String
from rostopic import ROSTopicHz
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from scipy import misc

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


def create_network_face_detection(gpu_memory_fraction):
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(gpu_memory_fraction / 2))
        # gpu_options = tf.GPUOptions(allow_growth=True)
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
                    if cv2.Laplacian(cropped, cv2.CV_64FC3).var() > 100:
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

    def __init__(self, model, classifier_location, data_dir, camera_topic, id, memory, pnet, rnet, onet):
        self._model = model
        self._data_dir = data_dir
        self._camera_topic = camera_topic
        self._id = id
        self._pnet = pnet
        self._rnet = rnet
        self._onet = onet

        self.changed = []

        self.classifiers = self.load_classifiers(classifier_location)
        self.classifier_dir = classifier_location
        self.predictions = []
        self.person_embs = np.load(os.path.join(classifier_location, 'embeddings.npy'))
        self.scheduler()

        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=(memory / 2))

        self._hz = int((get_hz(self._camera_topic) / 5) - 0.5)

        # Create the topic where the data will be published
        self.pub = rospy.Publisher("focus_vision/person/identification", personDataArray, queue_size=30)
        self.pub_image = rospy.Publisher("focus_vision/image/identification/" + self._id, Image, queue_size=30)
        self.pub_unknown_person = rospy.Publisher("focus_vision/image/unidentified", Image, queue_size=30)

        with tf.Graph().as_default():
            with tf.Session(
                    config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False)) as self.sess:
                facenet.load_model(self._model)
                # Retrieve the required input and output tensors for classification from the model
                self.images_placeholder = self.sess.graph.get_tensor_by_name("input:0")
                self.embeddings = self.sess.graph.get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = self.sess.graph.get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.bridge = CvBridge()

                # Subscribe to the camera topic provided in the args
                self.image_sub = rospy.Subscriber(self._camera_topic, Image, self.classify)

                # Subscribe to the topic providing the latest SVM classifier
                self.model_sub = rospy.Subscriber("/trained_model", String, self.reload_model)

                rospy.spin()

    def scheduler(self):
        thread_embeddings = threading.Thread(target=self.write_embeddings)
        thread_embeddings.daemon = True
        thread_embeddings.start()

        thread_reset = threading.Thread(target=self.reset_list)
        thread_reset.daemon = True
        thread_reset.start()

    def write_embeddings(self):
        while True:
            time.sleep(600)
            np.save(os.path.join(self.classifier_dir, 'embeddings.npy'), self.person_embs)

    def reset_list(self):
        while True:
            time.sleep(59)
            now = datetime.datetime.now().time()
            if datetime.time(now.hour, now.minute) == datetime.time(11, 59) \
                    or datetime.time(now.hour, now.minute) == datetime.time(23, 59):
                self.changed = []

    def publish_unknown(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, (160, 160))
        img = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
        self.pub_unknown_person.publish(self.bridge.cv2_to_imgmsg(img, 'bgr8'))

    def load_classifiers(self, folder):
        classifiers = {}
        for cls in os.listdir(folder):
            if '.pkl' in cls:
                with open(os.path.join(folder, cls), 'rb') as infile:
                    (model_pkl, class_names) = pickle.load(infile)
                    classifiers.update({class_names[0]: {'model': model_pkl, 'class_names': class_names}})
        return classifiers

    def reload_model(self, data):
        print(data.data)
        with open(data.data, 'rb') as infile:
            (model_pkl, class_names) = pickle.load(infile)
            if class_names[0] in self.classifiers:
                self.classifiers[class_names[0]]['model'] = model_pkl
                self.classifiers[class_names[0]]['class_names'] = class_names
            else:
                self.classifiers.update({class_names[0]: {'model': model_pkl, 'class_names': class_names}})
            self.person_embs = np.load(os.path.join(self.classifier_dir, 'embeddings.npy'))

    def predict(self, name, index, embedding):
        model = self.classifiers[name]['model']
        class_names = self.classifiers[name]['class_names']
        prediction = model.predict_proba([embedding])
        best_class_indices = np.argmax(prediction, axis=1)
        best_class_probabilities = prediction[np.arange(len(best_class_indices)), best_class_indices]
        for i in range(len(best_class_indices)):
            self.predictions.append({'index': index,
                                     'name': class_names[best_class_indices[i]],
                                     'confidence': best_class_probabilities[i]})

    def classify(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            return
        # if self._counter == self._hz:
        if self._counter == self._hz:
            start = time.time()
            self._counter = 0

            # Resize the image to a standard width and height
            cv_image = cv2.resize(cv_image, (1280, 720))
            # Change the color format of the image from BGR to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)

            # Align the faces in the list properly before classification

            images, boxes = align_data([cv_image], self.IMAGE_SIZE, self.MARGIN, self._pnet, self._rnet, self._onet)

            # Check if after aligning the list was returned and is not None
            if images is not None:
                feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)

                names = []
                for person in self.person_embs.item():
                    for i in range(len(emb_array)):
                        dist = np.sqrt(
                            np.sum(np.square(np.subtract(emb_array[i], self.person_embs.item().get(person)))))
                        if dist < 1.1 and person != 'Unknown':
                            poss = {'name': person, 'index': i, 'dist': dist}
                            names.append(poss)

                getIndex, getDist = lambda a: a['index'], lambda a: a['dist']  # or use operator.itemgetter
                groups = itertools.groupby(sorted(names, key=getIndex), key=getIndex)
                m = [min(b, key=getDist) for a, b in groups]
                namesCleaned = [l for l in names if l in m]

                if len(names) > 0:
                    threads = []
                    self.predictions = []
                    for i in range(len(namesCleaned)):
                        name = namesCleaned[i]
                        threads.append(threading.Thread(target=self.predict,
                                                        args=(name['name'], name['index'], emb_array[name['index']],)))
                        threads[i].start()
                        threads[i].join(0.15)

                msgs = personDataArray()
                found = False
                for i in range(len(boxes)):
                    for prediction in self.predictions:
                        predict_index = prediction['index']
                        if i == predict_index:
                            found = True
                            if prediction['confidence'] > 0.8 and prediction['name'] != 'Unknown':
                                self.person_embs.item()[prediction['name']] = emb_array[i]
                                msg = personData(prediction['name'].encode('ascii', 'ignore'), prediction['confidence'])
                                cv_image = cv2.rectangle(cv_image, (int(boxes[i][0]), int(boxes[i][1])),
                                                         (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                                cv_image = cv2.putText(cv_image, prediction['name'],
                                                       (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                       (0, 255, 0), 1, cv2.LINE_AA)
                            else:
                                """img = cv_image[int(boxes[i][1]):int(boxes[i][1] + (boxes[i][3] - boxes[i][1])),
                                      int(boxes[i][0]):int(boxes[i][0] + (boxes[i][2] - boxes[i][0]))]
                                self.publish_unknown(img)"""
                                msg = personData('Unknown', prediction['confidence'])
                                cv_image = cv2.rectangle(cv_image, (int(boxes[i][0]), int(boxes[i][1])),
                                                         (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                                cv_image = cv2.putText(cv_image, 'Unknown',
                                                       (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                                       (0, 255, 0), 1, cv2.LINE_AA)

                    if not found:
                        """img = cv_image[int(boxes[i][1]):int(boxes[i][1] + (boxes[i][3] - boxes[i][1])),
                              int(boxes[i][0]):int(boxes[i][0] + (boxes[i][2] - boxes[i][0]))]
                        self.publish_unknown(img)"""
                        msg = personData('Unknown', 1)
                        cv_image = cv2.rectangle(cv_image, (int(boxes[i][0]), int(boxes[i][1])),
                                                 (int(boxes[i][2]), int(boxes[i][3])), (0, 255, 0), 2)
                        cv_image = cv2.putText(cv_image, 'Unknown',
                                               (int(boxes[i][0]), int(boxes[i][1] - 10)),
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                               (0, 255, 0), 1, cv2.LINE_AA)

                    msgs.data.append(msg)
                self.pub.publish(msgs)
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
            self.pub_image.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
            print('Time to analyze frame: {} ms'.format((time.time() - start)*1000))
        else:
            self._counter += 1


if __name__ == "__main__":
    rospy.init_node("person_classification")

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file')
    parser.add_argument('classifier',
                        help='Path to the classifier (.pkl) files.')
    parser.add_argument('data_dir', type=str,
                        help='Path to the data directory containing classifier data.')
    parser.add_argument('camera_topic', type=str,
                        help='The ros topic name from the camera stream you want to analyze.')
    parser.add_argument('--id', type=str, help='The id of the camera being analized.', default='dlink_1')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.40)

    args = parser.parse_args()

    pnet, rnet, onet = create_network_face_detection(args.gpu_memory_fraction)

    classifier = PersonClassifier(args.model, args.classifier, args.data_dir, args.camera_topic, args.id,
                                  args.gpu_memory_fraction, pnet, rnet, onet)
