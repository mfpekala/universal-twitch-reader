from vid2frames import convert2frames
from preprocess import pre_process
from classify import classify
from process import process

# NOTE: Need to delete photos before
print("Converting video to frames... (Step 1 of 4)")
convert2frames(
    "data/videos/league/jankos.mp4",
    "data/images/jankos",
    fps=1/10 # Capture 1 frame every 10 seconds
)
"""
"""

# NOTE: Needs to be one folder out, so either make a dummy folder or do subset thing
# or just ignore (probably ignore)
print("Pre-processing frames... (Step 2 of 4)")
pre_process(
    "data/images",
    "data/boxes/jankos",
    conf_threshold=70,
    should_binarize=True,
    binarize_threshold=100,
    save_as_new_imgs=True
)
"""
"""

print("Classifying and labelling boxes... (Step 3 of 4)")
classifier, labeller = classify("data/boxes/jankos")
"""
"""

print("Processing Data... (Step 4 of 4)")
process("output/jankos", classifier, labeller)
"""
"""
