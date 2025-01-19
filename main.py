import os
from utils.video_utils import read_video,save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
import numpy as np
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from view_transformer.view_transformer import ViewTransformer
from speed_and_distance_estimator.speed_and_distance_estimator import SpeedDistanceEstimator

def main():
    video_path = os.path.join(os.getcwd(),'inputs/08fd33_4.mp4')
    frames = read_video(video_path)
    model_path = os.path.join(os.getcwd(),'models/best.pt')

    tracker = Tracker(model_path)
    tracks = tracker.get_obj_tracks(frames,read_from_stub=True,stub_path='stubs/track_stubs.pkl')

    tracks = tracker.add_position_to_tracks(tracks)

    cam_movement_estimator = CameraMovementEstimator(frames[0])
    cam_movement_per_frame = cam_movement_estimator.get_camera_movement(frames,True,stub_path='stubs/cam_movement_stubs.pkl')
    tracks = cam_movement_estimator.adjust_movement_to_tracks(tracks,cam_movement_per_frame)

    view_transformer = ViewTransformer()
    tracks = view_transformer.add_transformed_position_to_tracks(tracks)

    tracks['ball'] = tracker.interpolate_ball_positions(tracks['ball'])

    speed_distance_estimator = SpeedDistanceEstimator()
    tracks = speed_distance_estimator.add_speed_and_distance_to_tracks(tracks)

    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0],tracks['players'][0])

    for frame_no,player_track in enumerate(tracks['players']):
        for player_id,track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_no],track['bbox'],player_id)
            tracks['players'][frame_no][player_id]['team']=team
            tracks['players'][frame_no][player_id]['team_color']=team_assigner.team_colors[team]

    player_assigner = PlayerBallAssigner()
    team_ball_control = []

    for frame_no,player_track in enumerate(tracks['players']):
        ball_bbox = tracks['ball'][frame_no][1]['bbox']
        assigned_player = player_assigner.assign_ball_to_player(player_track,ball_bbox)

        if assigned_player!=-1:
            tracks['players'][frame_no][assigned_player]['has_ball']=True
            team_ball_control.append(tracks['players'][frame_no][assigned_player]['team'])
        else:
            if team_ball_control:
                team_ball_control.append(team_ball_control[-1])
            else:
                team_ball_control.append(1)
    
    team_ball_control = np.array(team_ball_control)

    output_video_frames = tracker.draw_annotations(frames,tracks,team_ball_control)

    output_video_frames = speed_distance_estimator.draw_speed_and_distance(output_video_frames,tracks)

    output_video_frames = cam_movement_estimator.draw_camera_movement(output_video_frames,cam_movement_per_frame)

    save_video(output_video_frames,'outputs/output_video.avi')

if __name__=="__main__":
    main()


'''
Detections(xyxy=array([[      905.8,      640.65,      972.22,      730.42],
       [     1337.4,      445.39,      1378.3,       518.6],
       [     541.03,      690.18,      579.75,      785.35],
       [     1317.5,      828.58,      1392.6,      922.65],
       [     893.35,      363.14,      921.44,      425.26],
       [     230.57,      516.78,      261.55,      596.36],
       [     1287.5,      396.34,      1322.7,      465.52],
       [     339.14,      494.24,      378.97,      570.84],
       [     1247.5,       432.3,      1278.7,      503.88],
       [     630.32,      597.67,       664.5,      681.15],
       [       1603,      629.49,      1640.5,      718.57],
       [     356.57,      735.11,      396.17,      839.98],
       [     373.33,      307.41,      399.36,      366.52],
       [     1010.3,       461.7,      1036.8,      529.56],
       [     1853.1,      808.79,      1893.1,      916.92],
       [     1168.7,      708.03,      1225.1,      808.53],
       [     1099.2,       308.2,      1123.2,      360.54],
       [     954.02,      225.51,      974.98,      275.99],
       [     1185.3,      356.87,      1211.9,      413.82],
       [     772.75,      369.94,      799.77,      431.68],
       [     314.58,      229.94,      331.65,      271.34],
       [     779.59,      422.44,      806.45,      485.67],
       [     1229.2,      885.95,      1245.2,      903.73],
       [     1906.9,      379.94,        1920,      445.94]], dtype=float32), mask=None, confidence=array([    0.92467,     0.92111,     0.91755,     0.91365,     0.91307,     0.90549,     0.90284,     0.90216,     0.90149,     0.90129,     0.90101,     0.90033,     0.89974,     0.89931,      0.8912,     0.89052,     0.88793,     0.86861,     0.86336,      0.8311,       0.822,     0.80047,     0.68705,     0.29445],
      dtype=float32), class_id=array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 2, 2, 2, 2, 3, 3, 2, 0, 1]), tracker_id=None, data={'class_name': array(['player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'player', 'referee', 'player', 'player', 'player', 'player', 'referee', 'referee', 'player', 'ball', 'goalkeeper'], dtype='<U10')}, metadata={})
'''