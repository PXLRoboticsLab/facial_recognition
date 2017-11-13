#!/usr/bin/env python
import argparse
import rospy
import time
import datetime
import numpy
from facial_recognition.msg import personData
from facial_recognition.msg import personDataArray
from pepper_controls.msg import Speech


def get_greeting(language, name):
    try:
        now = datetime.datetime.now().time()
        if datetime.time(7, 0) <= now <= datetime.time(10, 0):
            if language == "English":
                greeting = "Good morning"
            else:
                greeting = "Goedemorgen"
        elif datetime.time(13, 30) <= now <= datetime.time(16, 30):
            if language == "English":
                greeting = "Good afternoon"
            else:
                greeting = "Goedenmiddag"
        elif datetime.time(18, 0) <= now <= datetime.time(20, 0):
            if language == "English":
                greeting = "Good evening"
            else:
                greeting = "Goedenavond"
        else:
            if language == "English":
                greeting = "Hello"
            else:
                greeting = "Hallo"
    except:
        if language == "English":
            greeting = "Hello"
        else:
            greeting = "Hallo"

    if language == "English":
        greeting += " " + name.split(' ')[0] + ', welcome to pxl!'
    else:
        greeting += " " + name.split(' ')[0] + ', fijn om je hier te zien!'

    return greeting


class Output:
    _id = None
    _language = None
    _persons = {}
    _greeted = []

    def __init__(self, id, language):
        self._id = id
        self._language = language

        self.pub = rospy.Publisher('/focus_vision/pepper_robot/speech', Speech, queue_size=30)
        self.identification_sub = rospy.Subscriber('focus_vision/person/identification', personDataArray, self.callback)

    def callback(self, data):
        for person in data.data:
            if person.name in self._persons:
                self._persons[person.name]['confidence'].append(person.confidence)
                if len(self._persons[person.name]['confidence']) >= 3:
                    if len(person.name) != 32:
                        if person.id == self._id:
                            if numpy.average(self._persons[person.name][
                                              'confidence']) > 0.6 and person.name not in self._greeted:
                                print(get_greeting(self._language, person.name))
                                msg = Speech(self._language, get_greeting(self._language, person.name))
                                self.pub.publish(msg)
                                self._greeted.append(person.name)
                                self._persons[person.name]['confidence'] = [person.confidence]
                            else:
                                print('Threshold not met.')
                                print(person.name)
                                self._persons[person.name]['confidence'] = [person.confidence]
                        else:
                            print('Moving pepper to the correct camera location')
                    else:
                        if person.name not in self._greeted:
                            if self._language == "English":
                                print('I\'m sorry I don\'t seem to recognize you.')
                                msg = Speech(self._language, 'I don\'t seem to recognize you. May I ask for your name?')
                                self.pub.publish(msg)
                            else:
                                print('U lijk ik niet te herkennen sorry hiervoor')
                                msg = Speech(self._language, 'Sorry ik herken u helaas niet.')
                                self.pub.publish(msg)
                            self._greeted.append(person.name)
                        self._persons[person.name]['confidence'] = [person.confidence]
            else:
                self._persons.update(
                    {person.name: {'confidence': [person.confidence]}})
                print('False')


if __name__ == "__main__":
    rospy.init_node("classification_output")

    parser = argparse.ArgumentParser()
    parser.add_argument('id', type=str, help='The id of the camera subcribing to the analyzation topic')
    parser.add_argument('--language', type=str,
                        help='The language Pepper will speak.', default='English', choices=['English', 'Dutch'])

    args = parser.parse_args()

    output = Output(args.id, args.language)

    while True:
        time.sleep(5)
