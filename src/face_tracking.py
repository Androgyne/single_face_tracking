import cv2
import os
import torch
import argparse
import json
from face_detection import FaceDetector
from face_recognition import FaceRecognizer

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    image = cv2.imread(args.image_path)
    if image is None:
        print(f"Error: The target image at {args.image_path} is empty or could not be loaded.")
        return

    face_detector = FaceDetector(model_type='retinaface', device=device)
    face_recognizer = FaceRecognizer(target_image_path=args.image_path, device=device)

    # open video
    cap = cv2.VideoCapture(args.video_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    video_filename = os.path.splitext(os.path.basename(args.video_path))[0]
    print(f"Input video: {video_filename}")

    # create the output directory with the same name as input video
    output_dir = os.path.join(args.output_path, f"{video_filename}_clip")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    output_w = args.output_width
    output_h = args.output_height
    frame_cnt = 0
    recog_threshold = args.recog_threshold
    face_index = 1
    current_output_video_path = None
    face_video_out = None
    metadata = {"video_filename": args.video_path, "clips": []}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        timestamp = frame_cnt / fps
        frame_cnt += 1

        # detect faces in the current frame
        bboxes = face_detector.detect_faces(frame)

        best_similarity = float('inf')
        best_bbox = None 
        best_face_image = None

        # recognize faces in the detected bounding boxes
        for x1, y1, x2, y2 in bboxes:
            # crop the face area from the frame
            face_image = frame[y1:y2, x1:x2]
            face_embedding = face_recognizer.get_embedding(face_image)

            if face_embedding is not None:
                # compare the face embedding with the target face
                similarity = face_recognizer.compare_embeddings(face_embedding)
                print(f"similarity: {similarity}")

                if similarity < recog_threshold and similarity < best_similarity:
                    best_similarity = similarity
                    best_bbox = (x1, y1, x2, y2)
                    best_face_image = face_image

        if best_face_image is not None:

            # if this is the first frame with the target face, start a new video
            if not face_video_out:
                current_output_video_path = os.path.join(output_dir, f"{video_filename}_target_face_{face_index:03d}.mp4")
                face_video_out = cv2.VideoWriter(current_output_video_path, fourcc, fps, (output_w, output_h))
                print(f"Starting new video: {current_output_video_path}")
                face_index += 1
                metadata_clip = {"clip_filename": current_output_video_path, "start_timestamp": timestamp, "coordinates": []}

            # write the cropped face to the video
            output_face = cv2.resize(best_face_image, (output_w, output_h))
            face_video_out.write(output_face)
            coordinate = [int(best_bbox[0]), int(best_bbox[1]), int(best_face_image.shape[1]), int(best_face_image.shape[0])]
            metadata_clip["coordinates"].append(coordinate)
            

        else:
            if face_video_out:
                face_video_out.release() 
                face_video_out = None 
                print(f"Video saved at: {current_output_video_path}")

                metadata_clip["end_timestamp"] = timestamp
                metadata["clips"].append(metadata_clip)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break

    if face_video_out:
        face_video_out.release()
        print(f"Video saved at: {current_output_video_path}")

        metadata_clip["end_timestamp"] = timestamp
        metadata["clips"].append(metadata_clip)

    cap.release()
    cv2.destroyAllWindows()

    # save metadata as a JSON file in the output directory
    if metadata:
        metadata_file = os.path.join(output_dir, f"{video_filename}_metadata.json")
        with open(metadata_file, "w") as json_file:
            json.dump(metadata, json_file, indent=4)
        print(f"Metadata saved to {metadata_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face recognition on video.")
    
    parser.add_argument('--video_path', type=str, required=True, help="Path to the input video file.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the target face image.")
    parser.add_argument('--output_path', type=str, default="output/", help="Path to the output directory where clips and metadata will be saved.")
    parser.add_argument('--recog_threshold', type=float, default=0.6, help="Threshold for face recognition similarity.")
    parser.add_argument('--output_width', type=int, default=240, help="Width of the output video.")
    parser.add_argument('--output_height', type=int, default=320, help="Height of the output video.")

    args = parser.parse_args()

    main(args)
