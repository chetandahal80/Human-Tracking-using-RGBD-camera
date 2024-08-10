import numpy as np
import cv2

def linear_assignment(cost_matrix):
    try:
        import lap
        _, x, y = lap.lapjv(cost_matrix, extend_cost=True) # data assignment using Jonker-Volgenant algorithm
        return np.array([[y[i],i] for i in x if i >= 0]) 
    except ImportError:
        from scipy.optimize import linear_sum_assignment
        x, y = linear_sum_assignment(cost_matrix) 
        return np.array(list(zip(x, y)))


def cost_mat(bb_det, bb_trk, w_iou = 1, w_depth = 0):
    '''
    Computes IOU and reciprocal of depth distance between two bboxes in the form [x1,y1,x2,y2,d] and 
    generates cost matrix based on combined weight
    '''
    bb_trk = np.expand_dims(bb_trk, 0)
    bb_det = np.expand_dims(bb_det, 1)
    
    # cost associated with iou
    xx1 = np.maximum(bb_det[..., 0], bb_trk[..., 0])
    yy1 = np.maximum(bb_det[..., 1], bb_trk[..., 1])
    xx2 = np.minimum(bb_det[..., 2], bb_trk[..., 2])
    yy2 = np.minimum(bb_det[..., 3], bb_trk[..., 3])
    w = np.maximum(0., xx2 - xx1)
    h = np.maximum(0., yy2 - yy1)
    wh = w * h
    iou_matrix = wh / ((bb_det[..., 2] - bb_det[..., 0]) * (bb_det[..., 3] - bb_det[..., 1])                                      
        + (bb_trk[..., 2] - bb_trk[..., 0]) * (bb_trk[..., 3] - bb_trk[..., 1]) - wh)  

    # cost associated with depth
    d_det = bb_det[..., 4]
    d_trk = bb_trk[...,4]    
    d_matrix = np.abs(d_det - d_trk)
    epsilon = 1e-6
    reciprocal_dmatrix = 1/(d_matrix + epsilon) 
    max_reciprocal_distance = np.max(reciprocal_dmatrix)      
    normalized_reciprocal_dmatrix = reciprocal_dmatrix / max_reciprocal_distance

    cost = w_iou * iou_matrix + w_depth * normalized_reciprocal_dmatrix # combined cost matrix
                                 
    return(cost)  


def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2,depth] and returns z in the form
        [x,y,s,r,depth] where x,y is the centre of the box and s is the scale/area and r is
        the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    d = bbox[4]
    s = w * h    #scale is just area
    r = w / float(h)
    return (np.array([x, y, s, r, d], dtype = np.float32).reshape((5, 1)))


def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r,depth] and returns it in the form
        [x1,y1,x2,y2,depth] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,x[4]]).reshape((1,5))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,x[4],score]).reshape((1,6))


def detections_trackers_assignment(detections,trackers, w_iou, w_depth, iou_threshold = 0.3):
    '''
    Assigns detections to tracked object (both represented as bounding boxes)
    Returns 3 lists of matches, unmatched_detections and unmatched_trackers
    '''
    if(len(trackers)==0):
        return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)

    cost_matrix = cost_mat(detections, trackers, w_iou, w_depth) # generates cost matrix for data association

    if min(cost_matrix.shape) > 0:
        a = (cost_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-cost_matrix) # function call for data association
    else:
        matched_indices = np.empty(shape=(0,2))

    unmatched_detections = []
    for d, det in enumerate(detections):
        if(d not in matched_indices[:,0]):
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if(t not in matched_indices[:,1]):
            unmatched_trackers.append(t)

    #filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if(cost_matrix[m[0], m[1]]<iou_threshold):
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

class BBTracker():
    ''' Method to create Tracking object for each detected bounding box. Uses Kalman Filter technique'''
    count = 1
    def __init__(self, bbox):
        # initializing array for Kalman Filter
        self.kf = cv2.KalmanFilter(9,5)
        
        self.kf.transitionMatrix = np.array([[1,0,0,0,0,1,0,0,0],[0,1,0,0,0,0,1,0,0],[0,0,1,0,0,0,0,1,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,1],[0,0,0,0,0,1,0,0,0],[0,0,0,0,0,0,1,0,0],[0,0,0,0,0,0,0,1,0],[0,0,0,0,0,0,0,0,1]], dtype = np.float32)
        self.kf.measurementMatrix = np.array([[1,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0],[0,0,1,0,0,0,0,0,0],[0,0,0,1,0,0,0,0,0],[0,0,0,0,1,0,0,0,0]], dtype = np.float32)
        
        self.kf.measurementNoiseCov = np.eye(5, dtype = np.float32)
        self.kf.measurementNoiseCov[2:4, 2:4] *= 10.0
        
        self.kf.errorCovPost = np.eye(9, dtype = np.float32)
        self.kf.errorCovPost[5:, 5:] *= 1000.0
        self.kf.errorCovPost *= 10.0 

        self.kf.processNoiseCov = np.eye(9, dtype = np.float32)
        self.kf.processNoiseCov[7,7] *= 0.01
        self.kf.processNoiseCov[5:, 5:] *= 0.01

        self.time_since_update = 0
        self.kf.statePost = np.array([[bbox[0,0],bbox[1,0],bbox[2,0],bbox[3,0],bbox[4,0],0,0,0,0]], dtype = np.float32).T
        self.id = BBTracker.count
        BBTracker.count += 1
        self.history = []
        self.hit_streak = 0

    def predict(self):
        # prediction step of Kalman filter
        predicted = self.kf.predict()
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(predicted))
        return self.history[-1]
        
    def update(self, bbox):
        # correction step of Kalman filter
        self.time_since_update = 0
        self.history = []
        self.hit_streak += 1
        measurement = convert_bbox_to_z(bbox)
        self.kf.correct(measurement)

    def get_state(self):
        return(convert_x_to_bbox(self.kf.statePost))

class Tracker():
    ''' Tracker class for tracking the motion of each objects detected by YOLOv8 algorithm'''
    def __init__(self, min_iou = 0.3, min_streak = 3, max_age = 1, w_iou = 0.5, w_depth = 0.5):
        self.trackers = []
        self.min_iou = min_iou # minimum iou required to associate objects in successive frame
        self.min_streak = min_streak # minimum number of apperance required to be considered as detected object
        self.max_age = max_age # max age of an object to become obselete
        self.frame_count = 0 
        self.w_iou = w_iou # weight for iou
        self.w_depth = w_depth # weight for depth

    def tracking(self,detections = np.empty((0, 6))):
        ''' This method tracks the motion of each bounding box detected
            Input: detections = numpy array of size n_r*6 where columns denote x1, y1, x2, y2, depth, confidence
            Output: returns array containing information of tracker box '''

        trackers_list = np.zeros((len(self.trackers),6))
        to_del = []
        ret = []
        for i,track in enumerate(trackers_list):
            state = self.trackers[i].predict() # prediction stage of Kalman filter
            track[:] = [state[0,0], state[0,1], state[0,2], state[0,3], state[0,4], 0]
            if np.any(np.isnan(state)):
                to_del.append(i)
        trackers_list = np.ma.compress_rows(np.ma.masked_invalid(trackers_list))
        for t in reversed(to_del):
            self.trackers.pop(t)

        # data assiociation between detected objects and trackers
        matched, unmatched_detections, unmatched_trackers = detections_trackers_assignment(detections, trackers_list, self.w_iou, self.w_depth, self.min_iou)

        for index in matched:
            self.trackers[index[1]].update(detections[index[0]]) # correction step of Kalman filter for matched data

        for bbox in unmatched_detections:
            trkr = detections[bbox]
            trkr = convert_bbox_to_z(trkr)
            trk = BBTracker(trkr) # Kalman filter initialization for new detections or unmatched detections
            self.trackers.append(trk)

        length = len(self.trackers)
        for tracker in reversed(self.trackers):
            cur_state = tracker.get_state()[0]
            if ((tracker.time_since_update < 1) and ((tracker.hit_streak >= self.min_streak) or (self.frame_count <= self.min_streak))):
                ret.append(np.concatenate((cur_state,[tracker.id])).reshape(1,-1)) 
            
            length -= 1
        # remove dead tracklet
            if(tracker.time_since_update > self.max_age):
                self.trackers.pop(length)
        if(len(ret)>0):
            return np.concatenate(ret)
        return np.empty((0,5))           




        

        
             
            
            


