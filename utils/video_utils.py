import cv2 as cv

def read_video(video_path):
    vid = cv.VideoCapture(video_path)

    frames = []
    while True:
        ok,frame = vid.read()
        if not ok:
            break
        frames.append(frame)
    
    return frames

def save_video(video_frames,output_path):
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    out = cv.VideoWriter(output_path,fourcc,24,(video_frames[0].shape[1],video_frames[0].shape[0]))
    for frame in video_frames:
        out.write(frame)
    out.release()