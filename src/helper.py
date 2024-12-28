import numpy as np
import cv2
import os
import shutil
from zipfile import ZipFile

#global_bbox =  [{'frame': 0, 'object_id': 1, 'bbox': [np.float64(0.8098958333333334), np.float64(0.6199074074074075), np.float64(0.053125), np.float64(0.14351851851851852)]}, {'frame': 0, 'object_id': 2, 'bbox': [np.float64(0.7244791666666667), np.float64(0.5407407407407407), np.float64(0.034375), np.float64(0.1111111111111111)]}, {'frame': 1, 'object_id': 1, 'bbox': [np.float64(0.809375), np.float64(0.6189814814814815), np.float64(0.053125), np.float64(0.1398148148148148)]}, {'frame': 1, 'object_id': 2, 'bbox': [np.float64(0.7231770833333333), np.float64(0.5388888888888889), np.float64(0.033854166666666664), np.float64(0.10555555555555556)]}, {'frame': 2, 'object_id': 1, 'bbox': [np.float64(0.8080729166666667), np.float64(0.6171296296296296), np.float64(0.05364583333333333), np.float64(0.1361111111111111)]}, {'frame': 2, 'object_id': 2, 'bbox': [np.float64(0.7221354166666667), np.float64(0.5370370370370371), np.float64(0.034895833333333334), np.float64(0.09814814814814815)]}, {'frame': 3, 'object_id': 1, 'bbox': [np.float64(0.8080729166666667), np.float64(0.6171296296296296), np.float64(0.05364583333333333), np.float64(0.1361111111111111)]}, {'frame': 3, 'object_id': 2, 'bbox': [np.float64(0.7221354166666667), np.float64(0.5375), np.float64(0.0359375), np.float64(0.09907407407407408)]}]
#formatted_data=  [{'object': '1', 'class_id': '9'}, {'object': '2', 'class_id': '10'}]

#Function to extract all the frames of the video
def extract_all_frame(video_path, frame_dir):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while True:
        ret, frame = cap.read() #Read each frame
        if not ret:
            #If no more frame, break the loop
            break
        #Define a unique frame filename, like frame0.jpg, fram1.jpg, etc
        frame_filename = f"{frame_count}.jpg"
        frame_path = os.path.join(frame_dir, frame_filename)
       
        cv2.imwrite(frame_path, frame) # Save the frame as an image
        frame_count += 1
    #Release the video capture object
    print("extract successfully")
    cap.release()


def change_object_to_class_id(global_bbox, formatted_data):
    """
    Replaces the 'object_id' in global_bbox with the corresponding 'class_id' from formatted_data.

    Parameters:
    - global_bbox (list of dict): The list containing bounding box information.
    - formatted_data (list of dict): The list containing object to class_id mappings.

    Returns:
    - list of dict: The updated global_bbox with 'object_id' replaced by 'class_id'.
    """
    #mapping
    mapping = {item['object']: item['class_id'] for item in formatted_data}
    
    #Replace object_id with class_id
    for item in global_bbox:
        obj_id_str=str(item['object_id'])
        if obj_id_str in mapping:
            item['object_id']= mapping[obj_id_str]
        else:
            item['object_id']=None

    # Optionally convert to integers
    for item in global_bbox:
        if item['object_id'] is not None:
            item['object_id'] = int(item['object_id'])

    #for item in global_bbox:
    return global_bbox

#global_bbox = change_object_to_class_id(global_bbox, formatted_data)
#item after changing object's id:  
# [{'frame': 0, 'object_id': 9, 'bbox': [np.float64(0.8098958333333334), np.float64(0.6199074074074075), np.float64(0.053125), np.float64(0.14351851851851852)]},
#  {'frame': 0, 'object_id': 10, 'bbox': [np.float64(0.7244791666666667), np.float64(0.5407407407407407), np.float64(0.034375), np.float64(0.1111111111111111)]},
#  {'frame': 1, 'object_id': 9, 'bbox': [np.float64(0.809375), np.float64(0.6189814814814815), np.float64(0.053125), np.float64(0.1398148148148148)]},
#  {'frame': 1, 'object_id': 10, 'bbox': [np.float64(0.7231770833333333), np.float64(0.5388888888888889), np.float64(0.033854166666666664), np.float64(0.10555555555555556)]},
#  {'frame': 2, 'object_id': 9, 'bbox': [np.float64(0.8080729166666667), np.float64(0.6171296296296296), np.float64(0.05364583333333333), np.float64(0.1361111111111111)]},
#  {'frame': 2, 'object_id': 10, 'bbox': [np.float64(0.7221354166666667), np.float64(0.5370370370370371), np.float64(0.034895833333333334), np.float64(0.09814814814814815)]},
#  {'frame': 3, 'object_id': 9, 'bbox': [np.float64(0.8080729166666667), np.float64(0.6171296296296296), np.float64(0.05364583333333333), np.float64(0.1361111111111111)]}, {'frame': 3, 'object_id': 10, 'bbox': [np.float64(0.7221354166666667), np.float64(0.5375), np.float64(0.0359375), np.float64(0.09907407407407408)]}]


def convert_to_yolo_format(global_bbox):
    """
    Converts global_bbox data to YOLO format annotations.

    Parameters:
    - global_bbox (list of dict): The list containing bounding box information.

    Returns:
    - dict: A dictionary where keys are frame numbers and values are lists of YOLO annotation lines.
    """
    #Initialize a dictionary to hold annotations per frame
    annotations = {}
    
    for item in global_bbox:
        frame = item['frame']
        class_id = item['object_id']  # Already replaced with class_id
        bbox = item['bbox']  # [x_center, y_center, width, height], normalized
        
        # Ensure the frame key exists
        if frame not in annotations:
            annotations[frame] = []
        
        # Format: <class_id> <x_center> <y_center> <width> <height>
        yolo_annotation = f"{class_id} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}"
        annotations[frame].append(yolo_annotation)
    
    return annotations

#global_bbox = convert_to_yolo_format(global_bbox)
#print(global_bbox)
#{0: ['9 0.8098958333333334 0.6199074074074075 0.053125 0.14351851851851852', '10 0.7244791666666667 0.5407407407407407 0.034375 0.1111111111111111'],
#  1: ['9 0.809375 0.6189814814814815 0.053125 0.1398148148148148', '10 0.7231770833333333 0.5388888888888889 0.033854166666666664 0.10555555555555556'],
#  2: ['9 0.8080729166666667 0.6171296296296296 0.05364583333333333 0.1361111111111111', '10 0.7221354166666667 0.5370370370370371 0.034895833333333334 0.09814814814814815'],
#  3: ['9 0.8080729166666667 0.6171296296296296 0.05364583333333333 0.1361111111111111', '10 0.7221354166666667 0.5375 0.0359375 0.09907407407407408']}


def create_dataset_zip(yolo_data, frame_folder, project_name, output_zip_path="dataset.zip"):
    """
    Create a dataset zip file containing images and labels from specified frames.
    
    Args:
        yolo_data (dict): Dictionary with frame numbers as keys and YOLO annotations as values
        frame_folder (str): Path to the folder containing all frames
        output_zip_path (str): Path where the output zip file will be created
    """
    # Create absolute paths for temporary directories
    base_dir = os.path.dirname(frame_folder)
    temp_dir = os.path.join(base_dir, "temp_dataset")
    images_dir = os.path.join(temp_dir, "images")
    labels_dir = os.path.join(temp_dir, "labels")
    
    # Remove existing temp directory if it exists
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    
    # Create fresh directories
    os.makedirs(images_dir)
    os.makedirs(labels_dir)
    
    try:
        # Process each frame in yolo_data
        for frame_num, annotations in yolo_data.items():
            # Copy image file
            frame_path = os.path.join(frame_folder, f"{frame_num}.jpg")  # Assuming jpg format
            if os.path.exists(frame_path):
                shutil.copy2(frame_path, os.path.join(images_dir, f"{project_name}_{frame_num}.jpg"))
                print(f"Copied image: {project_name}_{frame_num}.jpg")
            else:
                print(f"Warning: Image {project_name}_{frame_num}.jpg not found")
            
            # Create label file
            label_path = os.path.join(labels_dir, f"{project_name}_{frame_num}.txt")
            with open(label_path, 'w') as f:
                for annotation in annotations:
                    f.write(f"{annotation}\n")
            print(f"Created label: {project_name}_{frame_num}.txt")
        
        # Create ZIP file with absolute path
        output_zip_path = os.path.join(base_dir, output_zip_path)
        with ZipFile(output_zip_path, 'w') as zipf:
            # Write all files from images directory
            for root, dirs, files in os.walk(images_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
            
            # Write all files from labels directory
            for root, dirs, files in os.walk(labels_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, temp_dir)
                    zipf.write(file_path, arcname)
                    
        print(f"ZIP file created at: {output_zip_path}")
        return True
    
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return False
    
    finally:
        # Clean up temporary directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print("Cleaned up temporary directory")