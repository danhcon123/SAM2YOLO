# 🚀 Using SAM2 🤖 to annotate video 🎞️ datas 🚀



# Instruction
---
▶️ Uploading Video ->  
▶️ Selecting Frame to Start Segmenting ->  
▶️ Segmenting through create and identify objects ->  
▶️ Pressing "Generating Maskes"-Button to check the generated maskes, if it is fitting expectation ->  
▶️ Propagating through video ->  
▶️ If Video not good, Choose a frame to edit the mask ->  
▶️ Generating new video ->  
▶️ Export in Video or YOLO-training data set format  


# Set up
---

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
    