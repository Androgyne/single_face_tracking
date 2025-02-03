import cv2
import torch
import argparse
from retinaface import RetinaFace 

class FaceDetector:
    def __init__(self, model_type='retinaface', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the face detector with the selected model type and device.
        
        :param model_type: The type of face detection model (currently supports 'retinaface')
        :param device: The device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        if model_type == 'retinaface':
            self.detector = RetinaFace
        else:
            raise ValueError("Unsupported model type: {}".format(model_type))

    def detect_faces(self, frame):
        """
        Detect faces in the given frame using the selected face detection model.
        
        :param frame: The input image frame (BGR format)
        :return: A list of bounding boxes for detected faces in the format [(x1, y1, x2, y2), ...]
        """
        # Convert the frame from BGR to RGB (required for RetinaFace)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.detector.detect_faces(frame_rgb)
        
        bboxes = []
        for face in faces.values():
            # Extract bounding box coordinates from the face dictionary
            x1, y1, x2, y2 = face['facial_area']
            bboxes.append((x1, y1, x2, y2))
        return bboxes

    def draw_bboxes(self, frame, bboxes):
        """
        Draw bounding boxes on the input frame for the detected faces.
        
        :param frame: The input image frame (BGR format)
        :param bboxes: A list of bounding boxes for detected faces in the format [(x1, y1, x2, y2), ...]
        :return: The frame with bounding boxes drawn around the faces
        """
        for (x1, y1, x2, y2) in bboxes:
            # draw a rectangle around the detected face
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # optional: Add a label "Face" above the bounding box
            cv2.putText(frame, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame


def main(args):
    """
    Unit test for the face detection functionality of the FaceDetector class.
    """
    # initialize face detector
    face_detector = FaceDetector(model_type='retinaface', device='cuda' if torch.cuda.is_available() else 'cpu')

    # load the image from the provided path
    frame = cv2.imread(args.image_path)

    if frame is None:
        print(f"Error: Could not read the image from {args.image_path}")
        return

    bboxes = face_detector.detect_faces(frame)

    frame_with_bboxes = face_detector.draw_bboxes(frame, bboxes)

    cv2.imshow("Detected Faces", frame_with_bboxes)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Detection from an image.")
    
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()

    main(args)
