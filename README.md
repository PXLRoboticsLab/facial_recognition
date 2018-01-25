# ROS - Facial recognition
This project is for easy face recognition on ROS camera streams.

## Getting started

### Prerequisites
This project requires you to install following dependecies to work.

* [Facenet](https://github.com/davidsandberg/facenet) 
* CUDA and cuDNN
* Tensorflow-gpu 1.0
* OpenCV 3
* ROS (Indigo or newer)

### Installation
Follow the steps below to install and run this project on your local machine.

* Clone this repository `$ git clone https://github.com/PXL-IT/facial_recognition.git`
* Give all the scripts run permission `$ sudo chmod u+x script_name.py`
* Start a roscore `$ roscore`
* Run the facial recognition `$ rosrun /path/to/model.pb /path/to/models/dir/ /path/to/aligend/data/ /ros/image/topic` optional arguments:
    * `--id` - **String** - The id of the camera in case you have multiple cameras.
    * `--gpu_memory_fraction` - **float** - Upper bound on the amount of GPU memory that will be used by the process.

## License
This project is licensed under the MIT License - see the [LICENSE.md](https://github.com/PXL-IT/SmartShop_Customer_Pipeline/blob/master/LICENSE.md) file for details.

## Authors
* [Maarten Bloemen](https://github.com/MaartenBloemen) 
