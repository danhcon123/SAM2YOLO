# 🚀 SAM2YOLO - Using SAM2 🤖 to generate labeled datas 🎞️ for training YOLO 🚀
  
  
## Introduction
  
In field of computer vision, the acceleration of data labeling remains a critical challenge. 🎯 To address this, my project leverages SAM2.1 from Meta to streamline the annotation process for video data. By allowing users to input a video, our software automatically separates it into individual frames.  

Users can then effortlessly annotate objects by simply pointing, enabling the generation of bounding boxes that are intelligently propagated across the entire video using SAM2.1's advanced segmentation capabilities.  

📹✨ The result is a set of accurately labeled images, ready to be used for training specialized object detection models with YOLO, significantly reducing the time and effort required for manual annotation. 🚀🔍  
  
  
  
## Workflow
  
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


## Installation

  
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
  
1. Create and activate virtual environment
```bash
python3 -m venv sam2

source sam2/bin/activate
```

2. Install [PyTorch](https://pytorch.org/get-started/locally/) based on your system's configuration. Match the installation with your CUDA Toolkit version if available. If CUDA is not installed, select the CPU-only version.

3. Install SAM2.1 by following the [SAM2.1](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) Installation Guide provided in the linked GitHub repository.

4. Clone this project's repository to match the structure of the project tree mentioned above:
```bash
git clone https://github.com/danhcon123/SAM2YOLO.git

cd SAM2YOLO
```

5. Install requirements
```bash
pip install -r requirements.txt
```

6. Set up project
 ```bash
python setup.py
```

## Run the project
                               
```bash
python app.py
```
