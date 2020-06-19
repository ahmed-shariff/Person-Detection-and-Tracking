#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""@author: ambakick
"""

import numpy as np
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from scipy.optimize import linear_sum_assignment

import helpers
import detector
import tracker
import cv2

# Global variables to be used by funcitons of VideoFileClop
frame_count = 0 # frame counter

max_age = 15  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers
# list for track ID
track_id_list= deque(['1', '2', '3', '4', '5', '6', '7', '7', '8', '9', '10'])

debug = False

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) 
    
    # Produces matches       
    # Solve the maximizing the sum of IOU assignment problem using the
    # Hungarian algorithm (also known as Munkres algorithm)
    
    matched_idx = linear_assignment(-IOU_mat)        

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0)
    
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       
    


class Pipeline():
    def __init__(self, tracker_list=tracker_list, max_age=max_age, min_hits=min_hits, track_id_list=track_id_list, debug=debug):
        self.det = detector.PersonDetector()
        self.frame_count = 0
        self.tracker_list = tracker_list
        self.max_age = max_age
        self.min_hits = min_hits
        self.track_id_list = track_id_list
        self.debug = debug

    def __call__(self, img):
        '''
        Pipeline function for detection and tracking
        '''

        self.frame_count+=1

        img_dim = (img.shape[1], img.shape[0])
        z_box = self.det.get_localization(img) # measurement
        if debug:
           print('Frame:', self.frame_count)

        x_box =[]
        if debug: 
            for i in range(len(z_box)):
               img1= helpers.draw_box_label(img, z_box[i], box_color=(255, 0, 0))
               plt.imshow(img1)
            plt.show()

        if len(self.tracker_list) > 0:
            for trk in self.tracker_list:
                x_box.append(trk.box)


        matched, unmatched_dets, unmatched_trks \
        = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3)  
        if debug:
             print('Detection: ', z_box)
             print('x_box: ', x_box)
             print('matched:', matched)
             print('unmatched_det:', unmatched_dets)
             print('unmatched_trks:', unmatched_trks)


        # Deal with matched detections     
        if matched.size >0:
            for trk_idx, det_idx in matched:
                z = z_box[det_idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk= self.tracker_list[trk_idx]
                tmp_trk.kalman_filter(z)
                xx = tmp_trk.x_state.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                x_box[trk_idx] = xx
                tmp_trk.box =xx
                tmp_trk.hits += 1

        # Deal with unmatched detections      
        if len(unmatched_dets)>0:
            for idx in unmatched_dets:
                z = z_box[idx]
                z = np.expand_dims(z, axis=0).T
                tmp_trk = tracker.Tracker() # Create a new tracker
                x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
                tmp_trk.x_state = x
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box = xx
                tmp_trk.id = self.track_id_list.popleft() # assign an ID for the tracker
                # print(tmp_trk.id)
                self.tracker_list.append(tmp_trk)
                x_box.append(xx)

        # Deal with unmatched tracks       
        if len(unmatched_trks)>0:
            for trk_idx in unmatched_trks:
                tmp_trk = self.tracker_list[trk_idx]
                tmp_trk.no_losses += 1
                tmp_trk.predict_only()
                xx = tmp_trk.x_state
                xx = xx.T[0].tolist()
                xx =[xx[0], xx[2], xx[4], xx[6]]
                tmp_trk.box =xx
                x_box[trk_idx] = xx


        # The list of tracks to be annotated  
        good_tracker_list =[]
        for trk in self.tracker_list:
            if ((trk.hits >= self.min_hits) and (trk.no_losses <=self.max_age)):
                good_tracker_list.append(trk)
                x_cv2 = trk.box
                if debug:
                    print('updated box: ', x_cv2)
                    print()
                img= helpers.draw_box_label(trk.id,img, x_cv2) # Draw the bounding boxes on the 
                                                 # images
        # Book keeping
        deleted_tracks = filter(lambda x: x.no_losses >self.max_age, self.tracker_list)  

        for trk in deleted_tracks:
            self.track_id_list.append(trk.id)

        self.tracker_list = [x for x in self.tracker_list if x.no_losses<=self.max_age]

        if debug:
           print('Ending tracker_list: ',len(self.tracker_list))
           print('Ending good tracker_list: ',len(good_tracker_list))

        cv2.imshow("frame",img)
        return img, good_tracker_list
    
if __name__ == "__main__":    
    
    pipeline = Pipeline()
    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        
        for i in range(len(images))[0:7]:
            image = images[i]
            image_box = pipeline(image)
            plt.imshow(image_box)
            plt.show()
           
    else: # test on a video file.
        
        # start=time.time()
        # output = 'test_v7.mp4'
        # clip1 = VideoFileClip("project_video.mp4")#.subclip(4,49) # The first 8 seconds doesn't have any cars...
        # clip = clip1.fl_image(pipeline)
        # clip.write_videofile(output, audio=False)
        # end  = time.time()
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 8.0, (640,480))

        while(True):
            
            ret, img = cap.read()
            #print(img)
            
            np.asarray(img)
            new_img, good_tracker_list = pipeline(img)
            # print([x.__dict__ for x in good_tracker_list])
            out.write(new_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        # print(round(end-start, 2), 'Seconds to finish')
