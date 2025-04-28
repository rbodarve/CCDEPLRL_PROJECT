# %%
import numpy as np
import argparse
import pickle
import cv2
import os
import time
import glob
from keras.models import load_model
from collections import deque
# %%
# Define paths based on CCDEPLRL_PROJECT structure
BASE_PATH = 'CCDEPLRL_PROJECT'
MODEL_PATH = os.path.join(BASE_PATH, 'model', 'violence_detection.h5')
INPUT_PATH = os.path.join(BASE_PATH, 'input')
OUTPUT_PATH = os.path.join(BASE_PATH, 'output')
# %%
def predict_video(video_path):
    # Make sure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    print(f"Processing video: {os.path.basename(video_path)}")
    
    # Load the model
    print("Loading model...")
    model = load_model(MODEL_PATH)
    
    # Initialize deque for prediction averaging
    Q = deque(maxlen=128)
    
    # Open the video
    vs = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = vs.get(cv2.CAP_PROP_FPS)
    frame_width = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    output_filename = os.path.join(OUTPUT_PATH, f"processed_{os.path.basename(video_path)}")
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(output_filename, fourcc, fps, (frame_width, frame_height))
    
    # Process each frame
    frame_count = 0
    violence_frames = 0
    
    while True:
        # Read the next frame
        (grabbed, frame) = vs.read()
        
        # If no frame was grabbed, we've reached the end
        if not grabbed:
            break
        
        # Create a copy of the frame for output
        output = frame.copy()
        
        # Preprocess the frame for prediction
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame = cv2.resize(processed_frame, (128, 128)).astype("float32")
        processed_frame = processed_frame / 255.0
        
        # Make prediction
        preds = model.predict(np.expand_dims(processed_frame, axis=0))[0]
        Q.append(preds)
        
        # Average predictions over time for stability
        results = np.array(Q).mean(axis=0)
        
        # Determine if frame shows violence (threshold > 0.5)
        is_violence = (results > 0.5)[0]
        
        # Count violent frames
        if is_violence:
            violence_frames += 1
        
        # Set text color based on prediction
        text_color = (0, 0, 255) if is_violence else (0, 255, 0)  # Red for violence, Green for non-violence
        
        # Display prediction on frame
        text = "Violence: {}".format(is_violence)
        cv2.putText(output, text, (35, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, text_color, 3)
        
        # Write the frame to output video
        writer.write(output)
        
        # Show the frame (optional, comment out for batch processing)
        # cv2.imshow("Processing", output)
        # key = cv2.waitKey(1) & 0xFF
        # if key == ord("q"):
        #    break
        
        frame_count += 1
    
    # Calculate violence percentage
    violence_percentage = (violence_frames / frame_count) * 100 if frame_count > 0 else 0
    
    # Release resources
    print(f"[INFO] Completed processing {os.path.basename(video_path)}")
    print(f"Violence percentage: {violence_percentage:.2f}%")
    writer.release()
    vs.release()
    
    return violence_percentage
# %%
def process_all_videos():
    # Get all videos from input directory
    video_files = []
    for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(INPUT_PATH, ext)))
    
    if not video_files:
        print("No video files found in the input directory!")
        return
    
    print(f"Found {len(video_files)} videos to process")
    
    # Process each video
    results = {}
    for video_path in video_files:
        video_name = os.path.basename(video_path)
        violence_percentage = predict_video(video_path)
        results[video_name] = violence_percentage
    
    # Write summary report
    report_path = os.path.join(BASE_PATH, 'violence_detection_report.txt')
    with open(report_path, 'w') as f:
        f.write("Violence Detection Summary Report\n")
        f.write("=" * 50 + "\n\n")
        f.write("Video Name | Violence Percentage\n")
        f.write("-" * 50 + "\n")
        
        for video_name, percentage in results.items():
            f.write(f"{video_name} | {percentage:.2f}%\n")
    
    print(f"Processing complete! Summary report saved to {report_path}")
# %%
if __name__ == "__main__":
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please run train.py first to create the model.")
        exit(1)
    
    # Create a command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--video", help="Path to a specific video (optional)")
    args = parser.parse_args()
    
    # Process single video if specified, otherwise process all videos in input folder
    if args.video:
        video_path = args.video
        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            exit(1)
        predict_video(video_path)
    else:
        process_all_videos()
