from moviepy.editor import *
import predict_module as M

def find_clip(video, groupname, favorite_name):
    parent_clip = VideoFileClip(video, audio=True)
    face_locations = []
    face_encodings = []
    face_names = []
    videoclip=[]
    audioclip=[]
    start_time = 0
    finish_time = 0
    flag= False
    while True:
        # Grab a single frame of video
        if start_time >= round(parent_clip.duration):
            print("break")
            break
        frame = parent_clip.get_frame(start_time)
        #print(frame)
        face_names, face_encodings = M.predict(frame, groupname='LE')
        if face_names == None:
            start_time += 0.5
            continue
        if favorite_name in face_names:
            videoclip.append(
                parent_clip.subclip(start_time, start_time + 0.5)
            )
        start_time += 0.5
    final_clip = concatenate_videoclips(videoclip)
    return final_clip



