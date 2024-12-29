# 🚀 SAM2YOLO - Using SAM2 🤖 to generate labeled datas 🎞️ for training YOLO 🚀


# Instruction

In the rapidly evolving field of computer vision, the acceleration of data labeling remains a critical challenge. 🎯 To address this, our project leverages SAM2.1 from Meta and YOLO from Ultralytics to streamline the annotation process for video data. By allowing users to input a video, our software automatically separates it into individual frames. Users can then effortlessly annotate objects by simply pointing, enabling the generation of bounding boxes that are intelligently propagated across the entire video using SAM2.1's advanced segmentation capabilities. 📹✨ The result is a set of accurately labeled images, ready to be used for training specialized object detection models with YOLO, significantly reducing the time and effort required for manual annotation. 🚀🔍


# Workflow

▶️ Uploading Video ->  
▶️ Selecting Frame to Start Segmenting ->  
▶️ Segmenting through create and identify objects ->  
▶️ Pressing "Generating Maskes"-Button to check the generated maskes, if it is fitting expectation ->  
▶️ Propagating through video ->  
▶️ If Video not good, Choose a frame to edit the mask ->  
▶️ Generating new video ->  
▶️ Export in Video or YOLO-training data set format.


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
    