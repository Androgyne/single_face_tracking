# Single Face Tracking

**Track** a target face in a video based on a reference image.

## Key Features

+ **Face Detection & Tracking**: Detect and track all faces in a video, focusing on identifying the target face based on a reference image.
+ **Continuous Target Face Clips**: Track the target face across frames and create continuous clips, ensuring no scene changes or full occlusions occur within each clip.
+ **Clip Splitting on Occlusion or Scene Change**: Automatically split clips if a scene change or full occlusion of the target face is detected.
+ **Cropped Video Clips**: Extract video segments containing only the target face, and save each segment as a separate video file.
+ **Metadata Generation**: Produce a metadata file (in JSON format) that includes:
    + File name of the cropped video.
    + Start and end timestamps of each clip.
    + Frame-by-frame face coordinates (e.g., [x, y, width, height]) of the target face.
  
## Environment Setup

### Clone the Repository
```bash
git clone https://github.com/Androgyne/single_face_tracking.git
cd single_face_tracking
```
### Create a Virtual Environment (Recommended)

```bash
conda create -n single_face_tracking python=3.11
conda activate single_face_tracking
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

Note: This project was tested on Google Colab using a T4 GPU, with CUDA version 12.5 and cuDNN version 9.2.1.

## Usage

<!-- The script provides a command-line interface for recognizing faces in a video and saving video clips where the target face appears. It also generates a metadata file that includes the start and end timestamps of each video clip, as well as the coordinates of detected faces. -->


To run the script, use the following command format:

```bash
python3 src/face_tracking.py --video_path <path_to_video> --image_path <path_to_face_image> --recog_threshold <threshold_value> --output_width <width> --output_height <height> --output_path <output_directory>
```
### Command-Line Arguments

The script accepts the following command-line arguments:

- `--video_path`: Path to the input video file (required).
  
- `--image_path`: Path to the target face image for recognition (required).

- `--recog_threshold`: Similarity threshold for face recognition (default: 0.6).
  
- `--output_width`: Width of the output video (default: 240).

- `--output_height`: Height of the output video (default: 320).

- `--output_path`: Path to save the output video clips and metadata (optional).




### Example

Here's an example of how to run the script with sample file paths:

```bash
python3 src/face_tracking.py --video_path sample/video_1.mp4 --image_path sample/image_1.jpg --recog_threshold 0.6 --output_width 240 --output_height 320 --output_path ./output
```

Note: In this repo, we provide three videos with their corresponding target images under `sample/`.


## Example Output

When using the CLI to process a video, the program will generate:

1. Cropped face video clips where the target face appears.
2. A metadata.json file with the following format:
```json
{
    "video_filename": "input_video.mp4",
    "clips": [
        {
            "clip_filename": "input_video_target_face_001.mp4",
            "start_timestamp": 12.345,
            "end_timestamp": 15.678,
            "coordinates": [
                [50, 100, 75, 100],
                [60, 110, 80, 100]
            ]
        },
        {
            "clip_filename": "input_video_target_face_002.mp4",
            "start_timestamp": 20.123,
            "end_timestamp": 25.456,
            "coordinates": [
                [40, 90, 70, 100]
            ]
        }
    ]
}

```

## Assumptions:

1.  Each frame contains only one target face.
2.  The video is relatively stable without extreme camera motion or rapid scene changes.
3.  The reference target face image is clear and represents the target face accurately.


## Limitations

1. **No Utilization of Temporal Information**: The system does not utilize temporal information between frames, which may lead to tracking failure when the target face undergoes gradual head pose changes.

2. **No Parallel Processing**: The system processes faces sequentially, limiting real-time performance and making it less efficient for videos with high frame rates or multiple faces.