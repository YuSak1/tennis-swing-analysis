from moviepy.editor import *
from moviepy.video.fx.mirror_x import mirror_x
import time
import os

# True for left handed players (Nadal)
lefty = False

def create_gif(input_path, output_path, lefty):
	# Read video file
	clip = VideoFileClip(input_path)
	# Video size
	clip = clip.resize(width=300)

	if lefty:
		clip = mirror_x(clip)

	clip.write_gif(output_path, fps=9)
	clip.close()


# for i in range(1,2):
# 	t_start = time.time() 
# 	# filename = "Nadal/Video_v2/Nadal" + str(i) + "_v2"
# 	filename = "Djokovic/Video_v3/Djokovic22_v3"
# 	input_path = "../videos/" + filename + ".mp4"
# 	output_path = "../videos/" + filename + ".gif"
 
# 	create_gif(input_path, output_path, lefty)

# 	t_end = time.time()
# 	elapsed_time = t_end - t_start
# 	print('Finished', filename)
# 	print('Elapsed time: {:.2f} minutes'.format((elapsed_time)/60))
# 	print('==========================================================')


# Get names of all files in the folder
input_folder = "../videos/Djokovic/feature_extraction/"
dir_list = os.listdir(input_folder)
for f in dir_list:
	t_start = time.time() 

	input_path = input_folder + f
	out_f = f[:-4] # remove extension
	output_path = "../videos/Djokovic/GIF_feature/" + out_f + ".gif"
 
	create_gif(input_path, output_path, lefty)

	t_end = time.time()
	elapsed_time = t_end - t_start
	print('Finished', out_f)
	print('Elapsed time: {:.2f} minutes'.format(elapsed_time/60))
	print('==========================================================')

print("Done!")
