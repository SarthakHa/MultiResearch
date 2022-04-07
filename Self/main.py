import os
from pathlib import Path
import numpy as np
import torch
from superglue import SuperGlue
from helpers import read_image
from matching import Matching

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

## Superglue
weight_to_use = 'indoor' #can also be outdoor
sinkhorn_iterations = 20 #this is default value
match_threshold = 0.2 #this is default value

## Superpoint
nms_radius = 4 #default
keypoint_threshold = 0.005 #default
max_keypoints = 1024 #default

## Non-model stuff
data_dir = 'Self_2/Self/assets/scannet_sample_images/' #This is default value
results_dir = 'dump_match_pairs/' #This is default value
resize = [640, 480] #default
resize_float = True #default
matches_path = '/dump_matches/'

config = {
        'superpoint': {
            'nms_radius': nms_radius,
            'keypoint_threshold': keypoint_threshold,
            'max_keypoints': max_keypoints
        },
        'superglue': {
            'weights': weight_to_use,
            'sinkhorn_iterations': sinkhorn_iterations,
            'match_threshold': match_threshold,
        }
    }

matching = Matching(config).eval().to(device)

data_dir = Path(data_dir)
print('Looking for data in directory \"{}\"'.format(data_dir))
results_dir = Path(results_dir)
results_dir.mkdir(exist_ok=True, parents=True)
print('Will write matches to directory \"{}\"'.format(results_dir))

name0, name1 = 'scene0711_00_frame-001680.jpg','scene0711_00_frame-001995.jpg' #change to match images inside of data directory
rot0,rot1 = 0,0 #default but can be provided - implement later
image0, inp0, scales0 = read_image(
    data_dir / name0, resize, rot0, resize_float)
image1, inp1, scales1 = read_image(
    data_dir / name1, resize, rot1, resize_float)
if image0 is None or image1 is None:
    print('Problem reading image pair: {} {}'.format(
        data_dir/name0, data_dir/name1))
    exit(1)

data = {'image0': inp0, 'image1': inp1} #goes into matching

# Perform the matching.
#pred = matching({'image0': inp0, 'image1': inp1})
pred = matching(data)
pred = {k: v[0].cpu().numpy() for k, v in pred.items()}
kpts0, kpts1 = pred['keypoints0'], pred['keypoints1']
matches, conf = pred['matches0'], pred['matching_scores0']

# Write the matches to disk.
out_matches = {'keypoints0': kpts0, 'keypoints1': kpts1,
                'matches': matches, 'match_confidence': conf}

np.savez(str(matches_path), **out_matches)