# %%
import os
import argparse
import sys
import subprocess
# %%
def print_header(message):
    print("\n" + "="*80)
    print(f" {message}")
    print("="*80 + "\n")
# %%
def setup_directories():
    """Setup the required directory structure if it doesn't exist"""
    # Define paths
    base_path = 'CCDEPLRL_PROJECT'
    directories = [
        os.path.join(base_path, 'dataset', 'Violence'),
        os.path.join(base_path, 'dataset', 'NonViolence'),
        os.path.join(base_path, 'input'),
        os.path.join(base_path, 'output'),
        os.path.join(base_path, 'model')
    ]
    
    # Create directories
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("Directory structure set up successfully!")
# %%
def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Violence Detection System")
    parser.add_argument("--setup", action="store_true", help="Setup directory structure")
    parser.add_argument("--extract", action="store_true", help="Extract frames from videos in dataset")
    parser.add_argument("--train", action="store_true", help="Train the violence detection model")
    parser.add_argument("--predict", action="store_true", help="Run prediction on videos in input folder")
    parser.add_argument("--video", help="Path to a specific video for prediction")
    
    args = parser.parse_args()
    
    # If no arguments, show help
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    # Setup directories if requested
    if args.setup:
        print_header("Setting up directories")
        setup_directories()
    
    # Extract frames if requested
    if args.extract:
        print_header("Extracting frames from videos")
        subprocess.run(["python", "extract_frame.py"])
    
    # Train model if requested
    if args.train:
        print_header("Training violence detection model")
        subprocess.run(["python", "train.py"])
    
    # Run prediction if requested
    if args.predict:
        print_header("Running prediction on videos")
        if args.video:
            subprocess.run(["python", "predict_video.py", "--video", args.video])
        else:
            subprocess.run(["python", "predict_video.py"])
# %%
if __name__ == "__main__":
    main()
