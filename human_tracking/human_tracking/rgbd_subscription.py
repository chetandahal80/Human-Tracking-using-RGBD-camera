import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
import numpy as np
from cv_bridge import CvBridge, CvBridgeError
import cv2
from ultralytics import YOLO
import cvzone
import math
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from message_filters import ApproximateTimeSynchronizer, Subscriber
from .state_estimation import *


class RGBDSubscriber(Node):
    def __init__(self):
        super().__init__('rgbd_subscriber')
        self.subscription_rgb = Subscriber(self, Image, '/camera/image_raw') # subscriber node to get rgb image
        self.subscription_depth = Subscriber(self, Image, '/camera/depth/image_raw') # subscriber node to get depth image
        self.ts = ApproximateTimeSynchronizer([self.subscription_rgb, self.subscription_depth], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.sync_callback)
        self.bridge = CvBridge()
        self.model = YOLO("src/human_tracking/human_tracking/Yolo-weights/yolov8l.pt") # YOLOv8 model for object detection
        self.classNames = list(self.model.names.values())
        self.latest_depth_image = None  # Variable to store the latest depth image
        self.tracker = Tracker(min_iou = 0.3, min_streak = 5, max_age = 20, w_iou = 0.5, w_depth = 0.5) # class initialization for object tracking 
    
    def get_depth(self, depth_image, bbox_arngd):
        ''' function to get the depth of objects detected
            Input: depth_image = array containing depth value of each pixel in a frame
                   bbox_arngd = dictionary containing bounding box number and its x1,y1,x2,y2,conf value
            Output: actual_depth = dictionary containg the depth of each object in the frame based on the depth values of 
                                   each pixel of bounding box'''
        depth_image = np.round(depth_image * (255/10.0)) # normalizing depth image
        depth_image[depth_image > 255] = 255
        depth_image = depth_image.astype(np.uint8)
        box_count = len(bbox_arngd)
        Th = 0.1 # Visibility threshold
        actual_depth = {}
        delta = 7.7
        mask = np.ones((depth_image.shape[0],depth_image.shape[1]))
        while bbox_arngd:
            j = 1
            Cj = {}
            for key in bbox_arngd:
                x1 = bbox_arngd[key][0]
                y1 = bbox_arngd[key][1]
                x2 = bbox_arngd[key][2]
                y2 = bbox_arngd[key][3]
                depth_values = depth_image[y1:y2, x1:x2] # extracting depth values for each bounding box
                roi_mask = mask[y1:y2, x1:x2] # creating region of interest for mask
                filter_depth = np.where(roi_mask == 1, depth_values, np.nan)
                valid_depth_values = filter_depth[~np.isnan(filter_depth)]
                filtered_depth = valid_depth_values[valid_depth_values <= 254]
                
                y, x, _ = plt.hist(filtered_depth, bins = 75, density = False) # histogram to obtain frequency of each depth value
                plt.close()
                min_count = np.min(y)
                max_count = np.max(y)
                y = (y - min_count)/(max_count - min_count) # normalized frequency 
                # obtaining Pd, Sd, F_pd and F_sd i.e. depth value with highgest and secound highest counts and there respective normalized frequency 
                peaks, _ = find_peaks(y, height = 0.025, distance = 16) 
                peak_values = y[peaks]
                sorted_order = np.argsort(peak_values)[::-1]
                sorted_indices = peaks[sorted_order]
                peak_values = np.sort(peak_values)[::-1]
                F_pd = peak_values[0]
                if (len(peak_values) > 1):
                    F_sd = peak_values[1]
                else:
                    F_sd = 0
                pd = x[sorted_indices[0]]

                V = F_pd - F_sd   # visibility calculation 
                if (V >= Th):
                    actual_depth[key] = pd * (10.0/255) 
                    condition = (depth_image >= (pd - delta)) & (depth_image <= (pd + delta))
                    mask[y1:y2, x1:x2][condition[y1:y2, x1:x2]] = 0
                else:
                    Cj[key] = bbox_arngd[key]
                    j = j + 1
            M = len(Cj)
            if (box_count == M):
                if(Th>0.01):
                    Th = Th/2
                else:
                    Th = 0
                bbox_arngd = Cj
                box_count = M
            if(box_count > M):
                bbox_arngd = Cj
                box_count = M
        print(actual_depth)
        return actual_depth


    def sync_callback(self, rgb_msg, depth_msg):
        # synchronized call back function for both rgb image and depth image subscriber nodes
        try:
            # Convert ROS Image message to OpenCV image
            img = self.bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth_image = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')

            self.latest_depth_image = np.array(depth_image, dtype=np.float32)
            img_copy = self.latest_depth_image.copy()
            results = self.model(img, stream=True) # obtaining object detection results using YOLOv8 model
            for r in results:
                box_depth = {}
                class_names = []
                bbox = {}
                state = []
                boxes = r.boxes

                # if no object is detected
                if (len(boxes) == 0):
                    print("No object detected")
                    break
                
                i = 1
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = x2 - x1, y2 - y1
                    conf = math.ceil((box.conf[0]) * 100) / 100
                    clss = int(box.cls[0])
                    if (conf > 0.4 and self.classNames[clss]=='person'):
                        cvzone.cornerRect(img, [x1, y1, w, h])
                        #cvzone.putTextRect(img, f'{self.classNames[clss]} {conf}', (max(0, x1), max(25, y1)), scale=1, thickness=1, offset=5)
                        # Extract depth information for the bounding box
                        if self.latest_depth_image is not None:
                            depth_values = self.latest_depth_image[y1:y2, x1:x2]
                            norm_depth_values = np.round(depth_values * (255/10.0))
                            flattened_array = norm_depth_values.flatten()
                            filtered_array = flattened_array[flattened_array <= 255]
                            box_depth[f'b{i}'] = filtered_array
                            bbox[f'b{i}'] = [x1, y1, x2, y2, conf]
                            class_names.append(self.classNames[clss])
                            i = i + 1
                # bounding box sorted as respected to the mean of all depth values of pixels within the box
                sorted_depth = {k: v for k, v in sorted(box_depth.items(), key=lambda item: np.mean(item[1]))}
                bbox_ordered = {key: bbox[key] for key in sorted_depth}

                depth = (self.get_depth(img_copy, bbox_ordered)) # function call to obtain actual depth of each objects 
                for key in bbox:
                    combined = bbox[key] + [depth[key]]
                    state.append(combined)
                state  = np.array(state)
                state[:,[4,5]] = state[:,[5,4]]

                results_tracker = self.tracker.tracking(state) # tracking the motion of each detected objects

                # creating the tracker box for each detected objects
                for result in results_tracker:
                    xx1, yy1, xx2, yy2, dd, Id = result
                    xx1, yy1, xx2, yy2, dd = int(xx1), int(yy1), int(xx2), int(yy2), round(dd, 2)
                    ww, hh = xx2-xx1, yy2-yy1
                    cvzone.cornerRect(img, [xx1, yy1, ww, hh], l = 9, rt = 2, colorR = (255,0,0))
                    cvzone.putTextRect(img, f'{Id} {dd}', (max(0, xx1), max(25, yy1)), scale=1, thickness=1, offset=5)          
                cv2.imshow("Image", img)
                cv2.waitKey(1)
        except CvBridgeError as e:
            self.get_logger().error(f'CvBridge Error: {str(e)}')
        except Exception as e:
            self.get_logger().error(f'An error occurred: {e}')


def main(args=None):
    rclpy.init(args=args)
    rgbd_subscriber = RGBDSubscriber()
    rclpy.spin(rgbd_subscriber)
    rgbd_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

