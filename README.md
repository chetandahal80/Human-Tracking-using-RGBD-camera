# Human-Tracking-using-RGBD-camera
Human detection using YOLOv8 alogrithm and tracking with rgb and depth information using Kalman filter

## Overview
This repository contains a ROS2 package which facilitates real-time human detection and tracking using a differential drive robot equipped with an RGB-D camera. The robot operates in a simulated environment within Gazebo, and the package relies on the YOLOv8 algorithm for human detection, with a Kalman filter for tracking the detected humans.

## Package Structure
1. description

   * Contains the URDF (Unified Robot Description Format) model of the robot, including sensor configurations. The URDF integrates the robot's differential drive system and the RGB-D camera, along with the necessary Gazebo plugins for simulating sensor data.

2. launch
   * launch_sim.launch.py: Launches the simulation environment, including Gazebo and the robot state publisher. It also spawns the robot into the Gazebo world defined in the worlds/ directory.
   * rsp.launch.py: Launches the Robot State Publisher, which publishes the robot's state information (e.g., joint states, TFs) based on the URDF.

3. human_tracking
   * rgbd_subscription.py: The core node that subscribes to the RGB and depth image topics, detects humans using YOLOv8, and calculates the depth of the detected objects. This node then uses a Kalman filter for tracking the detected humans over time.
   * state_estimation.py: Implements various functions and classes to support the Kalman filter-based tracking, including data association and bounding box processing.

## Main Functionalities
### Detection
* The rgbd_subscription.py node uses YOLOv8 to detect bounding boxes around humans in the RGB image stream.
* The bounding boxes are filtered by confidence score, and corresponding depth values are extracted from the depth image stream.

### Depth Estimation
* Depth values are processed within the bounding boxes to calculate an estimated depth for each detected object.
* The depth estimation algorithm filters outliers and uses histogram analysis to determine the most probable depth value for each object.

### Tracking
* A Kalman filter is implemented in state_estimation.py to predict and correct the position of the detected objects over time.
* Data association techniques (Jonker-Volgenant algorithm) are employed to match detected bounding boxes with existing tracks.

### Simulation
* The robot can be simulated in a Gazebo environment using the launch_sim.launch.py launch file.
* The RGB-D camera on the robot provides the necessary data streams for detection and tracking.

## Running the package
To run the package with a Gazebo simulation, run the following command line within the workspace after building the workspace: 
``` 
ros2 launch human_tracking launch_sim.launch.py world:=worlds/detects.world
```
This command will start the simulation environment with the robot and its sensors, ready to detect and track humans.

Next, we need to run the rgbd_subscription.py file using following command to start the human detection and tracking process:
``` 
ros2 run human_tracking rgbd
 ```
This command will start the nodes to subscribe rgb and depth image and performs the human detection and tracking operations.

Also, run the following command for teleoperation of the differential drive robot i.e. to give command velocity to the robot. 
```
ros2 run teleop_twist_keyboard teleop_twist_keyboard
```

## Dependencies 
The package relies on the following dependencies:
* rclpy: ROS2 Python client library.
  
* cv_bridge: For converting ROS image messages to OpenCV images.

* cv2: OpenCV (for image processing and Kalman filter implementation)"
  
* ultralytics: For using the YOLOv8 model.
  
* cvzone: For easy bounding box drawing and other utilities.
  
* scipy: For signal processing and data association.
  
* message_filters: For synchronizing RGB and depth image messages.

* matplotlib: For histogram computation during depth estimation
