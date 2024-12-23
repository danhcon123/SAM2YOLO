import os
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class Image_Segmentation:
    def __init__(self, model_config, checkpoint, image_dir, device="cuda"):
        """
        Initialize the SAM2 model for image segmentation and specify the image directory.
        
        :param model_config: Path to SAM2 model config file.
        :param checkpoint: Path to the trained model checkpoint file.
        :param image_dir: Directory where the images are stored (e.g., /home/gauva/flask_app/static/frames/).
        :param device: Device to run the model on (default is 'cuda').
        """
        self.device = device
        self.model = build_sam2(model_config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        self.image_dir = image_dir  # Directory for loading frames

    def load_image(self, image_name):
        """
        Load an image using its file name, from the predefined image directory.
        
        :param image_name: The file name of the image (e.g., 'frame_0.jpg').
        """
        image_path = os.path.join(self.image_dir, image_name)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image {image_name} not found in directory {self.image_dir}")
        
        image = Image.open(image_path).convert("RGB")
        self.image = np.array(image)
        self.predictor.set_image(self.image)

    def segment_image(self, input_points, input_labels, multimask_output=True):
        """
        Segment the image based on input prompts (points and labels).
        
        :param input_points: Coordinates of points for segmentation (list of (x, y)).
        :param input_labels: Labels for each point (1 for foreground, 0 for background).
        :param multimask_output: Whether to return multiple masks or a single best mask.
        :return: Segmentation masks, prediction scores, and logits.
        """
        input_points = np.array(input_points)
        input_labels = np.array(input_labels)
        
        # Perform prediction using SAM2
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=multimask_output
        )
        return masks, scores, logits

    def display_masks(self, masks, scores, input_points, input_labels):
        """
        Display segmentation masks on the image.
        
        :param masks: Segmentation masks from SAM2.
        :param scores: Confidence scores for each mask.
        :param input_points: Points used for segmentation (for visual display).
        :param input_labels: Labels of the points (for visual display).
        """
        for i, (mask, score) in enumerate(zip(masks, scores)):
            plt.figure(figsize=(10, 10))
            plt.imshow(self.image)
            self.show_mask(mask, plt.gca())
            self.show_points(input_points, input_labels, plt.gca())
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
            plt.axis('off')
            plt.show()

    def save_segmented_image(self, masks, save_dir="segmented_images"):
        """
        Save the segmented image with masks overlaid.
        
        :param masks: Segmentation masks from SAM2.
        :param save_dir: Directory to save the segmented image.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        for i, mask in enumerate(masks):
            image_with_mask = self.image.copy()

            # Overlay the mask on the image
            mask_colored = np.stack([mask] * 3, axis=-1) * np.array([255, 0, 0], dtype=np.uint8)
            image_with_mask[mask != 0] = mask_colored[mask != 0]

            save_path = os.path.join(save_dir, f'segmented_image_{i}.jpg')
            Image.fromarray(image_with_mask).save(save_path)
            print(f"Saved segmented image to {save_path}")

    @staticmethod
    def show_mask(mask, ax, borders=True):
        """
        Helper function to display mask on an image.
        
        :param mask: Binary mask to display.
        :param ax: Matplotlib axis to display the mask on.
        :param borders: Whether to show mask borders.
        """
        ax.imshow(np.ma.masked_where(mask == 0, mask), alpha=0.5, cmap='jet')

    @staticmethod
    def show_points(points, labels, ax):
        """
        Helper function to display input points on the image.
        
        :param points: Coordinates of the points.
        :param labels: Labels of the points (1 for foreground, 0 for background).
        :param ax: Matplotlib axis to display the points on.
        """
        points = np.array(points)  # Convert to NumPy array if necessary
        colors = ['red' if label == 1 else 'blue' for label in labels]
        ax.scatter(points[:, 0], points[:, 1], color=colors, s=100, edgecolor='white', linewidth=1.5)
        
    def clear_memory(self):
        """
        Free the GPU memory after task completion.
        """
        del self.model
        del self.predictor
        torch.cuda.empty_cache()
        print("Memory cleared.")

if __name__ == "__main__":
    model_cfg = "sam2_hiera_l.yaml"
    checkpoint_haha = "/home/gauva/sam2/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    # Initialize the Image_Segmentation class with the image directory
    FRAME_FOLDER = '/home/gauva/flask_app/static/frames/'
    segmenter = Image_Segmentation(model_cfg, checkpoint_haha, image_dir=FRAME_FOLDER, device="cuda")

    # Load an image by its name (e.g., frame_0.jpg)
    segmenter.load_image("0.jpg")

    # Segment the image with user-provided points and labels
    input_points = [(500, 600)]  # Example points
    input_labels = [1]  # Example labels (1 for foreground)
    masks, scores, logits = segmenter.segment_image(input_points, input_labels)

    # Display the masks
    segmenter.display_masks(masks, scores, input_points, input_labels)

    # Save the segmented image
    segmenter.save_segmented_image(masks)

    # Clear GPU memory
    segmenter.clear_memory()
