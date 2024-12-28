import os
import numpy as np
import torch
from PIL import Image, ImageDraw
from sam2.build_sam import build_sam2_video_predictor
from scipy.ndimage import binary_dilation
import io
import subprocess
import base64
import cv2
#from app import update_status, set_status

class VideoSegmentation:
    def __init__(self, model_cfg, checkpoint, video_dir, yolo_data, device=None):
        self.model_cfg = model_cfg
        self.checkpoint = checkpoint
        self.video_dir = video_dir
        self.yolo_data = yolo_data
    # Select device for computation
        if device:
            self.device = device
        else:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")

        print(f"Using device: {self.device}")

        if self.device.type == "cuda":
        # use bfloat16 for the entire notebook
            torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.device.type == "mps":
            print(
                "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
                "give numerically different outputs and sometimes degraded performance on MPS. "
                "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
            )
        # Load model
        self.predictor = build_sam2_video_predictor(self.model_cfg, self.checkpoint, device=self.device)

    def update_status(self, message):
        print(message)
        return message
    
    def init_inference_state(self):
        # Scan all the JPEG frame names in the video directory
        self.frame_names = [
            p for p in os.listdir(self.video_dir)
            if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg", ".png"]
        ]
        self.frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

        # Initialize inference state
        self.inference_state = self.predictor.init_state(video_path=self.video_dir)
        self.reset_state(self.inference_state)
        
    def reset_state(self, inference_state):
        self.predictor.reset_state(inference_state)

    def add_points(self, ann_frame_idx, ann_obj_id, points, labels):
    # Segment the input image
    # Add user-provided points to the interaction
        prompts = {}
        prompts[ann_obj_id] = points, labels
        #Generating the first masks
        _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=self.inference_state,
            frame_idx=ann_frame_idx,
            obj_id=ann_obj_id,
            points=points,
            labels=labels,
        )
        return out_obj_ids, out_mask_logits

    # Apply maskes to a image
    def apply_mask_on_image(self, image, mask, color=(255, 0, 0), alpha=0.3, border_thickness=5):
        if len(mask.shape) == 3:
            mask = np.squeeze(mask, axis=0)  # Removes the first dimension
        # Convert image to RGBA if it isn't already
        image = image.convert("RGBA")
        # Create an overlay image with the same size as the original image
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))  # Fully transparent overlay
        # Dilate the mask to make the border thicker
        if border_thickness > 0:
            mask = binary_dilation(mask, iterations=border_thickness)
        # Convert the mask to a binary format where 1s are the mask and 0s are the background
        mask_image = Image.fromarray((mask * 255).astype(np.uint8), mode="L")
        # Create a colored mask with the specified alpha value (transparency)
        colored_mask = Image.new("RGBA", image.size, color + (int(alpha * 255),))
        print(f"colored_mask: {colored_mask}")
        # Paste the colored mask into the overlay using the mask as a transparency guide
        overlay.paste(colored_mask, (0, 0), mask_image)
        # Composite the original image with the overlay (mask) using transparency
        result = Image.alpha_composite(image, overlay)
        
        # Bounding box storage
        non_zero_points = np.argwhere(mask > 0)

        # Calculate the bounding box coordinates from the mask
        if non_zero_points.size > 0:
            top_left = np.min(non_zero_points, axis=0)  # (y_min, x_min)
            bottom_right = np.max(non_zero_points, axis=0)  # (y_max, x_max)
            
            # Draw the bounding box on the image
            draw = ImageDraw.Draw(result)
            draw.rectangle(
                [(top_left[1], top_left[0]), (bottom_right[1], bottom_right[0])], 
                outline=color, width=border_thickness
            )
        return result
    
    
    #Apply Maskes to the whole video 
    def render_image_with_masks(self, frame_image, out_obj_ids, out_mask_logits, data):
        # Loop over the object IDs and their corresponding mask logits
        for i, out_obj_id in enumerate(out_obj_ids):
            mask = (out_mask_logits[i] > 0.0).cpu().numpy().squeeze()  # Convert mask to NumPy array
            # Find the corresponding object by its ID in the original data to get the correct color
            for original_obj in data:
                if original_obj['id'] == out_obj_id:
                    # Convert the hex color to an RGB tuple
                    hex_color = original_obj['borderColor'].lstrip('#')
                    color_rgb = tuple(int(hex_color[j:j+2], 16) for j in (0, 2, 4))
                    # Apply the mask with the corresponding color
                    frame_image = self.apply_mask_on_image(frame_image, mask, color=color_rgb)
                    break
        # Return the rendered image
        return frame_image
    
    def save_image_to_buffer(self, frame_image):
        # Save the result to a BytesIO object and encode it as base64
        buf = io.BytesIO()
        frame_image.save(buf, format="PNG")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        return img_base64
    
    def propagate_segmentation(self):
        # Propagate segmentation through the entire video
        self.video_segments = {}
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(self.inference_state):
            self.video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    def render_propagated_masks(self, data, display_video=False, video_name="output_video"):
        """Renders masks for each propagated frame, saves them in static/rendered_frames/,
            and creates an mp4 video after processing all frames.
            Also extract YOLO bbox data"""
        # Initialize YOLO data storage if not already done
        if not hasattr(self, 'yolo_data'):
            self.yolo_data = [] 
            
        # Directory to save rendered frames
        rendered_frames_dir = os.path.join("static", "rendered_frames")
        os.makedirs(rendered_frames_dir, exist_ok=True)
        
        # Init the video writer Set up the video writer (to be created after the first frame is processed)
        video_writer = None
        video_output_path = os.path.join("static/uploads", f"{video_name}_with_masked.avi")
        
        # Use a single window name for all frames
        window_name = "Video"
        # Clean the list yolo_data if there are any elements from objects in the first frame to the objects in the last frame
        try:
            first_frame = next(iter(self.video_segments.items()))[0]
            print(f"First frame number: {first_frame}")
            
            # Check if YOLO data exists and delete objects from the first frame onward
            if self.yolo_data is not None:
                self.del_objects_from_frame(first_frame)
                
        except Exception as e:
            print(f"Error: {e}")
            
        # Iterate over each frame and mask
        for frame_idx, obj_masks in self.video_segments.items():
            # Load the corresponding frame image using OpenCV
            frame_image_path = os.path.join(self.video_dir, f"{frame_idx}.jpg")
            frame_image = cv2.imread(frame_image_path)

            if frame_image is None:
                print(f"Error: Could not load frame image {frame_image_path}")
                continue  # Skip to the next frame if the image is not loaded
            
            # Initialize the video writer after loading the first frame
            if video_writer is None:
                frame_height, frame_width, _ = frame_image.shape
                video_writer = cv2.VideoWriter(video_output_path, cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (frame_width, frame_height))
            
            # Render the mask for the current frame
            for out_obj_id, mask in obj_masks.items():
                # Convert the mask from boolean to uint8 (required for OpenCV)
                mask_uint8 = (mask * 255).astype(np.uint8)  # Convert boolean mask to uint8

                # Remove the extra dimension if it exists
                if len(mask_uint8.shape) == 3 and mask_uint8.shape[0] == 1:
                    mask_uint8 = np.squeeze(mask_uint8, axis=0)  # Remove the first dimension
                    
                # Check if mask is valid
                if mask_uint8 is None or mask_uint8.size == 0:
                    print(f"Error: Invalid mask for frame {frame_idx}")
                    continue

                # Find the corresponding object in data to get the color
                for original_obj in data:
                    if original_obj['id'] == out_obj_id:
                        # Convert the hex color to RGB tuple
                        hex_color = original_obj['borderColor'].lstrip('#')
                        color_rgb = tuple(int(hex_color[i:i+2], 16) for i in [0, 2, 4])
                        darker_color_rgb = darken_color(color_rgb, factor=0.85)
                        color_bgr = (darker_color_rgb[2], darker_color_rgb[1], darker_color_rgb[0])  # Swap R and B
                        # Apply the mask on the frame using OpenCV
                        mask_resized = cv2.resize(mask_uint8, (frame_image.shape[1], frame_image.shape[0]))
                        colored_mask = np.zeros_like(frame_image)
                        colored_mask[mask_resized > 0] = color_bgr  # Apply the color to the mask areas
                        # Blend the mask with the frame
                        frame_image = cv2.addWeighted(frame_image, 1, colored_mask, 0.7, 0)

                        # --- Create Bounding Box ---
                        # Find the non-zero points in the mask
                        non_zero_points = np.argwhere(mask_resized > 0)
                        
                        if non_zero_points.size > 0:
                            # Get the coordinates for the bounding box
                            top_left = np.min(non_zero_points, axis=0)  # (y_min, x_min)
                            bottom_right = np.max(non_zero_points, axis=0)  # (y_max, x_max)
                            
                            # Draw the bounding box on the frame
                            cv2.rectangle(
                                frame_image, 
                                (top_left[1], top_left[0]),  # x_min, y_min
                                (bottom_right[1], bottom_right[0]),  # x_max, y_max
                                color_bgr,  # Green color for bounding box
                                5  # Thickness of 5
                            )
                            # Text: ID on the bounding box
                            object_id_text = f"{out_obj_id}"
                            text_position = (top_left[1], top_left[0] - 10) #Positioning above the box
                            text_size, _ = cv2.getTextSize(object_id_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)
                            text_width, text_height = text_size
                            cv2.rectangle(frame_image, (text_position[0], text_position[1]-text_height), # Top-left corner of the background rectangle
                                          (text_position[0] + text_width, text_position[1]+2), color_bgr, -1)  # Bottom-right corner of the background rectangle
                            cv2.putText(frame_image, object_id_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 2,
                                        (255,255,255), 3, cv2.LINE_AA)
                            # Convert bounding box to YOLO format
                            x_min = top_left[1]
                            y_min = top_left[0]
                            x_max = bottom_right[1]
                            y_max = bottom_right[0]

                            # Calculate YOLO coordinates
                            x_center = (x_min + x_max) / 2.0
                            y_center = (y_min + y_max) / 2.0
                            width = x_max - x_min
                            height = y_max - y_min

                            # Normalize
                            x_center_norm = x_center / frame_width
                            y_center_norm = y_center / frame_height
                            width_norm = width / frame_width
                            height_norm = height / frame_height
                                
                            # Assume out_obj_id is the class_id
                            self.yolo_data.append({
                                'frame': frame_idx,
                                'object_id': out_obj_id,
                                'bbox': [x_center_norm, y_center_norm, width_norm, height_norm]
                            })
                        break
            # Add black rectangle for background of the frame number
            cv2.rectangle(frame_image, (10, 10), (200, 80), (0, 0, 0), -1)
            
            # Add frame mumber text to the frame (top-left corner)
            cv2.putText(frame_image, f"Frame: {frame_idx}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255,255,255), 2, cv2.LINE_AA)
            
            #Save the rendered frame
            rendered_frames_path = os.path.join(rendered_frames_dir, f"{frame_idx}.jpg")
            cv2.imwrite(rendered_frames_path, frame_image)

            #Write the frame to the video file
            video_writer.write(frame_image)
            
            # Optionally display the frame
            if display_video:
                resized_frame = cv2.resize(frame_image, (640,480))
                cv2.imshow(window_name, resized_frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):  # Press 'q' to quit early
                    break
            
        #Release the video writer if it's initialized
        if video_writer is not None:
            video_writer.release()

        # Close all OpenCV windows if video was displayed
        if display_video:
            cv2.destroyAllWindows()        
        print(f"Video created at {video_output_path}")
        self.update_status("Prepairing rendered video")
        convert_avi_to_mp4(video_output_path)
    
    def group_by_frame(yolo_data):
        """
        Groups objects that have the same frame number together.

        Args:
            yolo_data (list): List of dictionaries containing frame information.

        Returns:
            dict: A dictionary where the key is the frame number and the value is a list of objects in that frame.
        """
        grouped_data = {}

        for entry in yolo_data:
            frame = entry.get('frame')
            if frame is None:
                raise ValueError("Missing 'frame' key in YOLO data entry")
            if frame not in grouped_data:
                grouped_data[frame] = []
            grouped_data[frame].append(entry)

        return grouped_data
        
    def get_yolo_data(self):
        return self.yolo_data

    def del_objects_from_frame(self, frame_id):
        """
        Deletes all objects and their bounding boxes from the specified frame_id to the end of the list.
        :param frame_id: The frame number from which to start deleting.
        """
        # Keep only frames with 'frame' < frame_id
        self.yolo_data = [entry for entry in self.yolo_data if entry['frame'] < frame_id]
        #print(f"yolo_data after got deleted from frame {frame_id}: ", self.yolo_data)
        #return self.yolo_data
    
    
def darken_color(color, factor=0.5):
    """
    Darken the given color by multiplying its RGB (or BGR) components by the factor.
    
    :param color: The original color as an (R, G, B) or (B, G, R) tuple.
    :param factor: The factor by which to darken the color (0.0 to 1.0).
                Lower values make the color darker.
    :return: Darkened color as an (R, G, B) or (B, G, R) tuple.
    """
    return tuple(max(0, int(c * factor)) for c in color)

def convert_avi_to_mp4(input_file = None):
    output_file = 'static/uploads/output.mp4'
    
    # Construct the FFMPEG command
    ffmpeg_command = [
        'ffmpeg', '-i', input_file,  # Input file
        '-c:v', 'libx264',           # H.264 codec for video
        '-crf', '23',                # Video quality
        '-preset', 'medium',         # Speed/size balance
        '-c:a', 'aac',               # AAC codec for audio
        '-b:a', '192k',              # Audio bitrate
        '-strict', 'experimental',   # Enable experimental AAC support
        output_file
    ]
    # Call FFMPEG using subprocess
    try:
        print("Starting video conversion...")
        subprocess.run(ffmpeg_command, check=True)
        print(f"MP4 video created: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion: {e}")


