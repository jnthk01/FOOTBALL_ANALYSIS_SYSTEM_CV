import pickle
import cv2 as cv
import numpy as np
import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance,measure_xy_distance
import os

class CameraMovementEstimator:
    def __init__(self,frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv.TermCriteria_EPS | cv.TermCriteria_COUNT,10,0.03)
        )
        first_frame_grayscale = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame_grayscale)
        mask_features[:,:20] = 1
        mask_features[:,900:1050] = 1

        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance =3,
            blockSize = 7,
            mask = mask_features
        )
    
    def adjust_movement_to_tracks(self,tracks,camera_movement_per_frame):
        for object_s,object_tracks in tracks.items():
            for frame_no,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    position = track_info['position']
                    camera_movement = camera_movement_per_frame[frame_no]
                    adjusted_position = (position[0]-camera_movement[0],position[1]-camera_movement[1])
                    tracks[object_s][frame_no][track_id]['position_adjusted'] = adjusted_position
        
        return tracks
                    

    def draw_camera_movement(self,frames,camera_movement):
        output_frames = []

        for frame_no,frame in enumerate(frames):
            frame = frame.copy()
            
            overlay = frame.copy()
            cv.rectangle(overlay,(0,0),(500,100),(255,255,255),-1)
            alpha = 0.6
            
            cv.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

            x_movement,y_movement = camera_movement[frame_no]

            frame = cv.putText(frame,f"Camera Movement X: {x_movement:.2f}",(10,30),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
            frame = cv.putText(frame,f"Camera Movement Y: {y_movement:.2f}",(10,60),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

            output_frames.append(frame)

        return output_frames

    def get_camera_movement(self,frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                return pickle.load(f)

        camera_movement = [[0, 0] for _ in range(len(frames))]

        old_gray = cv.cvtColor(frames[0],cv.COLOR_BGR2GRAY)

        old_features = cv.goodFeaturesToTrack(old_gray,**self.features)

        for frame_no in range(1,len(frames)):
            frame_gray = cv.cvtColor(frames[frame_no],cv.COLOR_BGR2GRAY)
            new_features,_,_ = cv.calcOpticalFlowPyrLK(old_gray,frame_gray,old_features,None,**self.lk_params)

            max_distance = 0
            camera_movement_x,camera_movement_y = 0,0

            for i,(new_feat,old_feat) in enumerate(zip(new_features,old_features)):
                new_features_points = new_feat.ravel()
                old_features_points = old_feat.ravel()

                distance = measure_distance(new_features_points,old_features_points)

                if distance>max_distance:
                    max_distance = distance
                    camera_movement_x,camera_movement_y = measure_xy_distance(new_features_points,old_features_points)
            
            if max_distance>self.minimum_distance:
                camera_movement[frame_no] = [camera_movement_x,camera_movement_y]
                old_features = cv.goodFeaturesToTrack(frame_gray,**self.features)

            old_gray = frame_gray.copy()
        
        if stub_path is not None:
            with open(stub_path,'wb') as f:
                pickle.dump(camera_movement,f)

        return camera_movement
