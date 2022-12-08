import face_recognition
import cv2
from moviepy.editor import *

import predict_module as M

input_movie = cv2.VideoCapture("../test_video/example6.mp4")
parent_clip = VideoFileClip('../test_video/example6.mp4', audio=True)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

print("please input name:")
favorite_name = input()

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
videoclip=[]
audioclip=[]
frame_number = 0
start_time = 0
finish_time = 0
second = round(input_movie.get(cv2.CAP_PROP_FPS))
print(second)
flag= False
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    frame_number += 1
    #print(int(input_movie.get(1)))
    if not ret:
        print("break")
        break
    rgb_frame = frame[:, :, ::-1]
    face_names, face_encodings = M.predict(frame)
    if face_names == None:
        continue
    if favorite_name in face_names:
        flag = True
        if int(input_movie.get(1))%second != 0:
            temp_time = int(input_movie.get(1))//second
            print(temp_time, int(input_movie.get(1)), frame_number)
            temp_time += 1
            frame_number = second * temp_time
            print(frame_number)
        input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    if int(input_movie.get(1))%second == 0:
        start_time += 1
        finish_time = start_time + 1
        if flag:
            videoclip.append(
                parent_clip.subclip(start_time, finish_time)
            )
            flag = False

# All done!
final_clip = concatenate_videoclips(videoclip)
final_clip.write_videofile("../test_video/moviepy9.mp4",  codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True)

input_movie.release()
cv2.destroyAllWindows()
