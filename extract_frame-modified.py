# %%
import cv2
import os
import glob
from tqdm import tqdm
# %%
# Define paths based on CCDEPLRL_PROJECT structure
BASE_PATH = 'CCDEPLRL_PROJECT'
DATASET_PATH = os.path.join(BASE_PATH, 'dataset')
PATH_violence = os.path.join(DATASET_PATH, 'Violence')
PATH_nonviolence = os.path.join(DATASET_PATH, 'NonViolence')
# %%
# Create directories to store extracted frames if they don't exist
os.makedirs(os.path.join(DATASET_PATH, 'frames', 'Violence'), exist_ok=True)
os.makedirs(os.path.join(DATASET_PATH, 'frames', 'NonViolence'), exist_ok=True)
# %%
# Extract frames from Violence videos
print("Extracting frames from Violence videos...")
for path in tqdm(glob.glob(PATH_violence + '/*')):
    fname = os.path.basename(path).split('.')[0]
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 5 == 0:  # Extract every 5th frame to reduce redundancy
            cv2.imwrite(os.path.join(DATASET_PATH, "frames", "Violence", "{}-{}.jpg".format(fname, str(count).zfill(4))), image)
        success, image = vidcap.read()
        count += 1
# %%
# Extract frames from NonViolence videos
print("Extracting frames from NonViolence videos...")
for path in tqdm(glob.glob(PATH_nonviolence + '/*')):
    fname = os.path.basename(path).split('.')[0]
    vidcap = cv2.VideoCapture(path)
    success, image = vidcap.read()
    count = 0
    while success:
        if count % 5 == 0:  # Extract every 5th frame to reduce redundancy
            cv2.imwrite(os.path.join(DATASET_PATH, "frames", "NonViolence", "{}-{}.jpg".format(fname, str(count).zfill(4))), image)
        success, image = vidcap.read()
        count += 1

print("Frame extraction completed!")
