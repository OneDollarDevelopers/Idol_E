from moviepy.editor import *
import predict_module as M

parent_clip = VideoFileClip('../test_video/example1.mp4', audio=True)


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

print(parent_clip.duration)
flag= False
while True:
    # Grab a single frame of video
    if start_time >= round(parent_clip.duration):
        print("break")
        break
    frame = parent_clip.get_frame(start_time)
    #print(frame)
    face_names, face_encodings = M.predict(frame, groupname='Loona')
    if face_names == None:
        start_time += 0.5
        continue

    if favorite_name in face_names:
        print(start_time)
        videoclip.append(
            parent_clip.subclip(start_time, start_time + 0.5)
        )
    start_time += 0.5


# All done!
final_clip = concatenate_videoclips(videoclip)
final_clip.write_videofile("../test_video/moviepy1.mp4",  codec='libx264', 
                     audio_codec='aac', 
                     temp_audiofile='temp-audio.m4a', 
                     remove_temp=True)


