# 🚀 SAM2YOLO - Using SAM2 🤖 to generate labeled datas 🎞️ for training YOLO 🚀
  
  
# Introduction
  
  
In field of computer vision, the acceleration of data labeling remains a critical challenge. 🎯 To address this, my project leverages SAM2.1 from Meta to streamline the annotation process for video data. By allowing users to input a video, our software automatically separates it into individual frames.  

Users can then effortlessly annotate objects by simply pointing, enabling the generation of bounding boxes that are intelligently propagated across the entire video using SAM2.1's advanced segmentation capabilities.  

📹✨ The result is a set of accurately labeled images, ready to be used for training specialized object detection models with YOLO, significantly reducing the time and effort required for manual annotation. 🚀🔍  
  
  
  
# Workflow
  
  
▶️ Uploading Video ->  
▶️ Selecting Frame from seperated frames from video ->  
▶️ Select the detected objects by placing a pointer on them ->  
▶️ Press the 'Generate Masks' button to review the generated masks and verify they meet your expectations ->  
▶️ Propagate the masks through the video ->  
▶️ If one or more labeled frames are incorrect, select a frame to edit the mask ->  
▶️ Generate new labeled data ->  
▶️ Export the results in YOLO training dataset format. 
```bash
project_name_dataset/
├── images/
    └──...
    └── project_name_300.jpg
├── label/
    └── ...
    └── project_name_300.txt
```
  
  
# Installation

  
Project tree
```bash
project/
├── SAM2YOLO/
    └──...
    └── app.py
├── sam2/
    └── ...
    └── checkpoints/
        └── sam2.1_hiera_large.pt
```

Install requirements

```bash
pip install -r requirements.txt
```

Follow the installation instructions for SAM2 provided in the official repository:

https://github.com/facebookresearch/sam2"

# TODO:
---

## Frontend: 

Home: == Upload

Mask_Generating
+ Loading screen
+ Showed message + Effect
+ Id update after the first loop
+ The show coordination

Result:
+ Export button -> Yolo format (Boundingbox + Mask (Segmenting))

## Backend:

Mask_Generating
+ Showed message 

Result:
+ The rendered video
+ The showed rendered image should also include the image that's not been segmenting (the not looping through images)
+ Export data method for training Yolo format (Boundingbox + Mask (Segmenting))
    + Name of data
    + Choosing export format
    + Create File + export
    + Download button
    