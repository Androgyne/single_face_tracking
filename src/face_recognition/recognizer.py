from facenet_pytorch import InceptionResnetV1, MTCNN
import torch
import numpy as np
from scipy.spatial.distance import cosine
import cv2

class FaceRecognizer:
    def __init__(self, target_image_path=None, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the FaceRecognizer class.
        
        :param target_image_path: (str) Path to the target image for face recognition. Default is None.
        :param device: (str) Device to run the model on ('cuda' or 'cpu'). Default is 'cuda' if a GPU is available.
        """
        self.facenet = InceptionResnetV1(pretrained='vggface2').eval().to(device)  # move FaceNet to GPU

        # if a target image is provided, extract its embedding
        self.device = device
        if target_image_path:
            target_image = cv2.imread(target_image_path)
            self.target_size = target_image.shape
            target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
            self.target_embedding = self.get_embedding(target_image)
            
            

    def get_embedding(self, face_image):
        """
        Process the face image and extract its embedding using FaceNet.

        :param face_image: (np.array) The face image to extract the embedding from.
        :return: (torch.Tensor) The embedding of the face image.
        """
        # preprocess the face image (resize and normalize)
        face_image_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
        face_image_rgb = cv2.resize(face_image_rgb, (224, 224))
        face_tensor = torch.tensor(face_image_rgb).permute(2, 0, 1).float()  # HWC to CHW
        face_tensor = face_tensor.unsqueeze(0).to(self.device) 
        
        # normalize to [-1, 1] range as FaceNet expects input
        face_tensor = (face_tensor / 255.0 - 0.5) * 2.0

        # extract face embedding
        with torch.no_grad():
            face_embedding = self.facenet(face_tensor)

        return face_embedding

    def compare_embeddings(self, face_embedding):
        """
        Compare the given face embedding with the target face embedding using cosine similarity.

        :param face_embedding: (torch.Tensor) The embedding of the face to compare with the target embedding.
        :return: (float) Cosine similarity between the target embedding and the provided face embedding.
        """
        # compare the embeddings using cosine similarity
        similarity = cosine(self.target_embedding[0].cpu().numpy(), face_embedding.cpu()[0].numpy())  # Move tensors to CPU before comparison
        return similarity
