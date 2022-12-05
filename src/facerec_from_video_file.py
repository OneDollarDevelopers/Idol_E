import face_recognition
import cv2
from moviepy.editor import *

import predict_module as M

input_movie = cv2.VideoCapture("../test_video/example3.mp4")
parent_clip = VideoFileClip('../test_video/example3.mp4', audio=True)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))


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
while True:
    # Grab a single frame of video
    ret, frame = input_movie.read()
    second = input_movie.get(cv2.CAP_PROP_FPS)
    #print(length)
    frame_number += 1

    # Quit when the input video file ends
    if not ret:
        print("break")
        break

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_names, face_encodings = M.predict(frame)
    if face_names == None:
        
        continue
    flag= True
    if favorite_name in face_names:
        flag=True
        print("Nice!")
    else:
        flag=False
        print("Sad")
        
    print("Writing frame {} / {}".format(frame_number, length))
    start_time = (frame_number-1) * (1 / second)
    finish_time = start_time + (1/second)
    if flag:
        videoclip.append(
            parent_clip.subclip(start_time, finish_time)
        )
        print('Nice!') 
    else:
        continue

# All done!
final_clip = concatenate_videoclips(videoclip)
final_clip.write_videofile("../test_video/moviepy.mp4",  codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True)

input_movie.release()
cv2.destroyAllWindows()
