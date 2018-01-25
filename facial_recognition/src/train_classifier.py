#!/usr/bin/env python
import tensorflow as tf
import rospy
import facenet
import argparse
import pickle
import time
import os
import numpy as np
from sklearn.svm import SVC
from std_msgs.msg import String


def read_paths_from_folder(folder, paths):
    nrof_paths = 0
    for img in os.listdir(folder):
        if '.png' in img:
            nrof_paths += 1
            paths.append(os.path.join(folder, img))

    return nrof_paths


class TrainClassifier:
    BATCH_SIZE = 1000

    def __init__(self, args):
        self.folder = args.data_dir
        self.output = args.output_dir
        self.model = args.model
        self.topic = args.topic
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)

        try:
            self.person_embs = np.load(os.path.join(self.output, 'embeddings.npy')).item()
        except IOError:
            self.person_embs = {}

        with tf.Graph().as_default():
            with tf.Session(
                    config=tf.ConfigProto(gpu_options=self.gpu_options, log_device_placement=False)) as self.sess:
                print('Loading feature extraction model')
                facenet.load_model(self.model)

                self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                self.embedding_size = self.embeddings.get_shape()[1]

                self.pub = rospy.Publisher("trained_model", String, queue_size=1)

                self.sub = rospy.Subscriber(self.topic, String, self.callback)

                rospy.spin()

    def callback(self, data):
        print('Training classifier(s)')
        for person in os.listdir(self.folder):
            if person != 'Unknown':
                if not os.path.exists(os.path.join(self.output, '{}.pkl'.format(person))) \
                        or time.time() - os.path.getmtime(os.path.join(self.output, '{}.pkl'.format(person))) \
                                > (30 * 24 * 60 * 60):
                    print('Creating classifier for: {}'.format(person))
                    paths = []
                    nrof_paths = read_paths_from_folder(os.path.join(self.folder, person), paths)
                    labels = [0] * nrof_paths
                    nrof_paths = read_paths_from_folder(os.path.join(self.folder, 'Unknown'), paths)
                    labels += [1] * nrof_paths

                    classifier_filename_exp = os.path.expanduser(
                        os.path.join(self.output, '{}.pkl'.format(person)))

                    images = facenet.load_data(paths, False, False, 160)
                    feed_dict = {self.images_placeholder: images, self.phase_train_placeholder: False}
                    emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)

                    model = SVC(kernel='linear', probability=True)
                    model.fit(emb_array, labels)

                    name = person.replace('_', ' ')
                    class_names = [name, 'Unknown']

                    with open(classifier_filename_exp, 'wb') as outfile:
                        pickle.dump((model, class_names), outfile)

                    if name in self.person_embs:
                        self.person_embs[name] = emb_array[0]
                    else:
                        self.person_embs.update({name: emb_array[0]})

                    np.save(os.path.join(self.output, 'embeddings.npy'), self.person_embs)

                    print('Saved classifier model to file "%s"' % classifier_filename_exp)
                    self.pub.publish(classifier_filename_exp)


if __name__ == '__main__':
    rospy.init_node('classifier_training')

    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str,
                        help='Path to a model protobuf (.pb) file.')
    parser.add_argument('data_dir', type=str,
                        help='Path to dir containing the aligned person photos.')
    parser.add_argument('output_dir', type=str,
                        help='Path to where the classifier (.pkl) files will be saved.')
    parser.add_argument('--topic', type=str,
                        help='The ros topic to listen to.', default='/train_command')
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=0.20)

    args = parser.parse_args()

    trainer = TrainClassifier(args)
