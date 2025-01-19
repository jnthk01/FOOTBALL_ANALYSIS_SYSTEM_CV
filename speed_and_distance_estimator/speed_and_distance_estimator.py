import cv2 as cv
import sys
sys.path.append('../')
from utils.bbox_utils import measure_distance,get_foot_position
class SpeedDistanceEstimator:
    def __init__(self):
        self.frame_window = 5
        self.frame_rate = 24

    def add_speed_and_distance_to_tracks(self,tracks):
        total_distance = {}

        for object_s,object_tracks in tracks.items():
            if object_s=='ball' or object_s=='referees':
                continue

            no_of_frames = len(object_tracks)
            for frame_no in range(0,no_of_frames,self.frame_window):
                last_frame_no = min(frame_no+self.frame_window,no_of_frames-1)
                for track_id,_ in object_tracks[frame_no].items():
                    if track_id not in object_tracks[last_frame_no]:
                        continue
                    
                    start_position = object_tracks[frame_no][track_id]['position_transformed']
                    end_position = object_tracks[last_frame_no][track_id]['position_transformed']
                    
                    if start_position is None or end_position is None:
                        continue

                    distance = measure_distance(start_position,end_position)
                    time_elapsed = (last_frame_no-frame_no)/self.frame_rate

                    speed_metres_per_second = distance/time_elapsed
                    speed_km_per_hour = speed_metres_per_second*3.6

                    if object_s not in total_distance:
                        total_distance[object_s] = {}

                    if track_id not in total_distance[object_s]:
                        total_distance[object_s][track_id] = 0
                    
                    total_distance[object_s][track_id]+=distance

                    for frame_num_batch in range(frame_no,last_frame_no):
                        if track_id not in tracks[object_s][frame_num_batch]:
                            continue
                        tracks[object_s][frame_num_batch][track_id]['speed'] = speed_km_per_hour
                        tracks[object_s][frame_num_batch][track_id]['distance'] = total_distance[object_s][track_id]
        
        return tracks

    def draw_speed_and_distance(self,video_frames,tracks):
        output_frames = []

        for frame_no,frame in enumerate(video_frames):
            for object_s,object_tracks in tracks.items():
                if object_s=='ball' or object_s=='referees':
                    continue
            
                for _,track_info in object_tracks[frame_no].items():
                    if 'speed' in track_info:
                        speed = track_info.get('speed',None)
                        distance = track_info.get('distance',None)

                        if speed is None or distance is None:
                            continue

                        bbox = track_info['bbox']
                        position = get_foot_position(bbox)
                        position = list(position)
                        position[1]+=40

                        position = tuple(map(int,position))

                        cv.putText(frame,f"{speed:.2f} km/h",position,cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                        cv.putText(frame,f"{distance:.2f} m",(position[0],position[1]+20),cv.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2)
                
            output_frames.append(frame)

        return output_frames