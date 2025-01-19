import os
import sys 
import pickle
import cv2 as cv
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO
sys.path.append('../')
from utils.bbox_utils import get_bbox_width,get_center_bbox,get_foot_position

class Tracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    def add_position_to_tracks(self,tracks):
        for object,object_tracks in tracks.items():
            for frame_no,track in enumerate(object_tracks):
                for track_id,track_info in track.items():
                    bbox = track_info['bbox']
                    if object == 'ball':
                        position = get_center_bbox(bbox)
                    else:
                        position = get_foot_position(bbox)
                    tracks[object][frame_no][track_id]['position'] = position

        return tracks

    def interpolate_ball_positions(self,ball_positions):
        ball_positions = [x.get(1,{}).get('bbox',[]) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        df_ball_positions = df_ball_positions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [{1: {"bbox":x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions

    def detect_frames(self,frames):
        batch_size=20 
        detections = [] 
        for i in range(0,len(frames),batch_size):
            detections_batch = self.model.predict(frames[i:i+batch_size],conf=0.1)
            detections += detections_batch
        return detections

    def get_obj_tracks(self,frames,read_from_stub=False,stub_path=None):

        if read_from_stub and stub_path is not None and os.path.exists(os.path.join(os.getcwd(),stub_path)):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            return tracks

        detections = self.detect_frames(frames=frames)

        tracks={
            "players":[],
            "referees":[],
            "ball":[]
        }

        for frame_no,detection in enumerate(detections):
            class_names = detection.names
            class_names_inv = {j:i for i,j in class_names.items()}

            detection_sv = sv.Detections.from_ultralytics(detection)

            # Converting goal keeper to player
            for obj_ind,cls_id in enumerate(detection_sv.class_id):
                if class_names[cls_id]=='goalkeeper':
                    detection_sv.class_id[obj_ind] = class_names_inv["player"]

            # Tracking objects
            detection_with_tracks = self.tracker.update_with_detections(detection_sv)
            
            # storing, tracking id with respective bounding box
            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_det in detection_with_tracks:
                bounding_boxes = frame_det[0].tolist()
                cls_id = frame_det[3]
                track_id = frame_det[4]

                if cls_id==class_names_inv['player']:
                    tracks["players"][frame_no][track_id] = {"bbox":bounding_boxes}

                if cls_id==class_names_inv['referee']:
                    tracks["referees"][frame_no][track_id] = {"bbox":bounding_boxes}

            for frame_det in detection_with_tracks:
                bounding_boxes = frame_det[0]
                cls_id = frame_det[3]

                if cls_id==class_names_inv['ball']:
                    tracks["ball"][frame_no][1] = {"bbox":bounding_boxes} # Since there is only one ball no need track_id
            
            if stub_path is not None:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks,f)
        
        return tracks

    def draw_ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center,_ = get_center_bbox(bbox=bbox)
        width = get_bbox_width(bbox=bbox)

        cv.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv.LINE_4
        )

        # On ellipse place a rectange with their track_id
        rect_width = 40
        rect_height = 20
        x1_rect = x_center-(rect_width//2)
        x2_rect = x_center+(rect_width//2)
        y1_rect = (y2-(rect_height//2))+15
        y2_rect = (y2+(rect_height//2))+15

        if track_id is not None:
            cv.rectangle(frame,(x1_rect,y1_rect),(x2_rect,y2_rect),color,cv.FILLED)

            x1_text = x1_rect+12
            if track_id>99:
                x1_text-=10
            
            cv.putText(frame,f"{track_id}",(int(x1_text),int(y1_rect+15)),cv.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)

        return frame

    def draw_triangle(self,frame,bbox,color):
        y1 = bbox[1]
        x,_ = get_center_bbox(bbox)

        tri_points = np.array([[x,y1],[x-10,y1-20],[x+10,y1-20]],dtype=np.int32)
        tri_points = tri_points.reshape((-1, 1, 2))

        cv.drawContours(frame,[tri_points],0,color,cv.FILLED)
        cv.drawContours(frame,[tri_points],0,(0,0,0),2)

        return frame
    
    def draw_team_control(self,frame,frame_no,team_ball_control):
        overlay = frame.copy()

        cv.rectangle(overlay,(1350,850),(1900,970),(255,255,255),-1)
        alpha = 0.4
        cv.addWeighted(overlay,alpha,frame,1-alpha,0,frame)

        team_ball_control_till_frame = team_ball_control[:frame_no+1]
        team_1 = team_ball_control_till_frame[team_ball_control_till_frame==1].shape[0]
        team_2 = team_ball_control_till_frame[team_ball_control_till_frame==2].shape[0]
        team1_frames = team_1/(team_1+team_2)
        team2_frames = team_2/(team_1+team_2)

        cv.putText(frame,f"Team 1 Ball Control: {team1_frames*100:.2f}%",(1400,900),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)
        cv.putText(frame,f"Team 2 Ball Control: {team2_frames*100:.2f}%",(1400,950),cv.FONT_HERSHEY_SIMPLEX,1,(0,0,0),3)

        return frame
    
    def draw_annotations(self,frames,tracks,team_ball_control):
        output_video_frame = []
        for (frame_no,frame) in enumerate(frames):
            frame = frame.copy()

            player_dict = tracks['players'][frame_no]
            referees_dict = tracks['referees'][frame_no]
            ball_dict = tracks['ball'][frame_no]

            for track_id,player in player_dict.items():
                color = player.get('team_color',(0,0,255))
                frame = self.draw_ellipse(frame,player['bbox'],color,track_id)

                if player.get("has_ball",False):
                    frame = self.draw_triangle(frame,player['bbox'],(0,0,255))

            for _,referee in referees_dict.items():
                frame = self.draw_ellipse(frame,referee['bbox'],(0,255,255))
            
            for _,ball in ball_dict.items():
                frame = self.draw_triangle(frame,ball['bbox'],(0,255,0))

            frame = self.draw_team_control(frame,frame_no,team_ball_control)

            output_video_frame.append(frame)
        
        return output_video_frame

'''
Detections(xyxy=array([[     989.64,      659.67,      1044.6,      750.78],
       [     1618.2,      654.98,      1670.7,       749.3],
       [     540.46,      702.92,      600.45,      798.09],
       [     238.36,      519.47,      273.78,      601.56],
       [     374.09,      311.16,       400.1,      371.73],
       [     677.23,      608.44,      713.77,      692.88],
       [     915.34,      366.61,      940.18,      429.42],
       [     1373.2,      449.75,      1414.7,      522.75],
       [     1211.5,      357.74,      1241.4,      416.84],
       [     1112.2,      314.78,      1146.5,      363.23],
       [     1308.2,      397.94,      1337.8,      469.19],
       [     358.47,      500.64,      386.22,      575.76],
       [     1853.7,      806.68,      1902.3,      915.25],
       [     336.91,       738.8,      377.62,      846.08],
       [     781.54,       425.9,      813.57,      498.36],
       [     1027.8,      467.62,      1058.8,      536.65],
       [     962.96,       227.8,      984.66,      277.67],
       [     1272.6,      433.48,      1295.4,      504.61],
       [     1229.3,      830.85,        1281,      927.01],
       [     1245.5,      760.38,      1291.3,      840.22],
       [      318.7,      227.11,      340.37,      270.96],
       [     777.97,      375.12,       802.6,      435.96],
       [     1243.9,      717.11,      1285.2,      792.69],
       [     1245.7,      727.55,      1294.2,      811.03]], dtype=float32), mask=None, confidence=array([    0.93676,     0.92261,     0.92077,     0.91514,     0.91009,     0.90884,     0.90415,     0.90393,     0.90337,     0.90258,     0.89573,     0.89551,     0.89128,     0.89106,     0.89029,     0.88954,     0.88481,     0.88223,     0.87741,     0.82489,     0.82301,     0.80471,     0.77041,     0.13729],
      dtype=float32), class_id=array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 2, 2, 2, 3, 3, 2, 2]), tracker_id=array([ 1,  9,  7, 10, 11,  5,  6,  3, 18, 25, 12,  4, 16, 14, 13,  2, 27, 15,  8, 28, 20, 17, 22, 19]), data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'referee', 'player', 'player'], dtype='<U7')}, metadata={})
'''