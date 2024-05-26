from moviepy.editor import *
# from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.fx.mirror_x import mirror_x


def gif(input_file, output_path, lefty=False):
    # Read video file
    clip = VideoFileClip(input_file)
    # Video size
    clip = clip.resize(width=200)

    # mirror if lefty
    if lefty:
        clip = mirror_x(clip)

    clip.write_gif(output_path, fps=9, loop=0)
    clip.close()
    print("GIF is created.")

    return None
