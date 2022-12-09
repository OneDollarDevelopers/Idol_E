   continue

    if favorite_name in face_names:
        print(start_time)
        videoclip.append(
            parent_clip.subclip(start_time, start_time + 1)
        )
    start_time += 1
