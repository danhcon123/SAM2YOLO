#app für sam2
from flask import Flask, render_template, request, send_file, url_for, redirect, jsonify, send_from_directory,session

import os
import cv2
from video_segmentation import VideoSegmentation
import numpy as np
from PIL import Image
import torch
import atexit


app = Flask(__name__)
app.secret_key = '017643771350'
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#-------------------------Video upload function----------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
 
#Set the foleder where uploaded files will be saved
UPLOAD_FOLDER = '/home/gauva/flask_app/static/uploads/'
FRAME_FOLDER = '/home/gauva/flask_app/static/frames/'
RENDERED_FRAME_FOLDER = '/home/gauva/flask_app/static/rendered_frames/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize the VideoSegmentation object globally
status_message = ""
model_cfg = "sam2_hiera_l.yaml"
checkpoint = "/home/gauva/sam2/segment-anything-2/checkpoints/sam2_hiera_large.pt"
video_dir = FRAME_FOLDER
global_objects = [] #Stores the object button for the not first time propagation
global_iteration = 0
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024 #1GB
global_bbox=[]

#Allowed file extensions (videos)
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webp', 'm4v'}
 
#Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
 
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

#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#---------------------------------------Webpages---------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
 
@app.errorhandler(413)
def request_entity_too_large(error):
    return "File is too large. The limit is 100MB.", 413
 
# Route for the homepage
@app.route('/')
def home():
    return render_template('upload.html')
 
# Route for the about page
@app.route('/upload', methods = ['GET', 'POST'])
def upload(): #Upload video
    if request.method == 'POST':
        #Check if the POST request has the file part
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
       
        file = request.files['file']
       
        # If no file is selected
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
       
        #If file is allowed and has an appropiate extension
        if file and allowed_file(file.filename):
            #session['status'] = "Uploading video and separating frames"
            set_status("Uploading video and separating frames...")
            update_status()
            filename = file.filename
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path) #Save the file to the uploaded folder
           
            # Clear the frames directory before saving new frames
            if os.path.exists(FRAME_FOLDER):
                for old_file in os.listdir(FRAME_FOLDER):
                    file_path = os.path.join(FRAME_FOLDER, old_file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                       
            #Extract and save all the frames from the upload video    
            extract_all_frame(video_path, FRAME_FOLDER)
            return jsonify({'status': 'complete', 'video_filename': filename})
       
    # If it's a GET request, render the about page with the uploaded video
    video_filename = request.args.get('video_filename')
    return render_template('upload.html', video_filename=video_filename)
 
@app.route('/choosing_frame')
def choosing_frame():
    #Get list of all saved frames to display them in the choosing_frame page
    frames = os.listdir(FRAME_FOLDER)
    frames = sorted([f for f in frames if f.endswith('.jpg')], key=lambda x: int(x[:-4])) #Filter to only include .jpg image
    # Serve the annotating page (choosing_frame.html) with the first frame of the video
    return render_template('choosing_frame.html', frames=frames)
 
@app.route('/generating_mask')
def generating_mask():
    chosen_frame = session.get('chosen_frame') #Get the frame from the session
    print("chosen frame: ", chosen_frame)
    file_exists = os.path.exists(os.path.join(RENDERED_FRAME_FOLDER, chosen_frame))
    if not chosen_frame:
        return "No frame chosen!", 400
    return render_template('generating_mask.html', filename=chosen_frame, file_exists=file_exists)

@app.route('/result')
def result():
        #Get list of all saved frames to display them in the choosing_frame page
    frames = os.listdir(RENDERED_FRAME_FOLDER)
    frames = sorted([f for f in frames if f.endswith('.jpg')], key=lambda x: int(x[:-4])) #Filter to only include .jpg image
    return render_template('result.html', frames=frames)


#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------SEND_RECEIVE_SERVER/ CLIENT---------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
 
#Serve individual frame images for annotation page
@app.route('/frames/<filename>')
def get_frame(filename):
    return send_from_directory(FRAME_FOLDER, filename)

#Serve individual frame images for annotation page
@app.route('/rendered_frames/<filename>')
def get_rendered_frame(filename):
    return send_from_directory(RENDERED_FRAME_FOLDER, filename)

#To handle the choice of frame and store the frame number
@app.route('/choose_frame', methods=['POST'])
def choose_frame():
    #Get the selected frame number from the request
    data = request.json
    selected_frame = request.json.get('frame')
    session['chosen_frame'] = selected_frame #Store the chosen frame in session
    # Log the received frame to verify
    print(f"Received frame: {selected_frame}")
    return jsonify({'status': 'success', 'chosen_frame': selected_frame})
 
#Save positive and negative coordinates to use SAM2 on it
@app.route('/generate_mask', methods=['POST'])
def generate_mask():
    global global_bbox
    video_segmenter = VideoSegmentation(model_cfg, checkpoint, video_dir, global_bbox)
    video_segmenter.init_inference_state()
    data = request.json   
    # Check if the data contains at least one object
    if not data or len(data) == 0:
        return jsonify({'status': 'error', 'message': 'No objects provided'}), 400    
    # Track the current frame index and load the image only once per frame
    current_frame_idx = None
    frame_image = None
    # hold all the clicks we add for visualization
    prompts = {} 
    # Loop through each object in the objectsList
    for obj in data:
        ann_obj_id = obj['id']  # Object ID
        points_data = obj['points']  # Points list
        labels_data = obj['labels']  # Labels list
        frames = obj['frames']  # Frame name(s)
        # Assuming the frame name format is 'i.jpg', extract the frame index (i)
        ann_frame_idx = int(frames[0].split('.')[0])

        # Load the frame only if it's not already loaded
        if current_frame_idx != ann_frame_idx:
            frame_image = Image.open(os.path.join(video_dir, f"{ann_frame_idx}.jpg"))
            current_frame_idx = ann_frame_idx
            
        # Convert points and labels to NumPy arrays
        points = np.array([[int(p['x']), int(p['y'])] for p in points_data], dtype=np.float32)
        labels = np.array(labels_data, dtype=np.int32)
        # Store the points and labels for this object
        prompts[ann_obj_id] = points, labels
        # Call the add_points method for the current object
        set_status("Generating masks")
        update_status()
        out_obj_ids, out_mask_logits = video_segmenter.add_points(
            ann_frame_idx=ann_frame_idx,
            ann_obj_id=ann_obj_id,
            points=points,
            labels=labels
        )
        # Render the image with masks using the VideoSegmentation class method
        frame_image = video_segmenter.render_image_with_masks(frame_image, out_obj_ids, out_mask_logits, data)
    # Save the rendered image to a buffer and return the base64 image
    img_base64 = video_segmenter.save_image_to_buffer(frame_image)
    # Free GPU memory after the task is done
    del video_segmenter
    torch.cuda.empty_cache()
    return jsonify({'status': 'success','message': 'All objects processed successfully!', 'image_data': img_base64})

        
@app.route('/propagate_segmentation', methods=['POST'])
def propagate_segmentation():
    global global_bbox
    global global_objects

    # Path to the file to delete
    output_file_path = os.path.join('static', 'uploads', 'output.mp4')
    
    # Check if the file exists and delete it
    if os.path.exists(output_file_path):
        os.remove(output_file_path)
        print(f"Deleted file: {output_file_path}")
    else:
        print(f"File not found: {output_file_path}")

    global_objects = []  # Reset the global_objects array at the start of each iteration
    #global video_segmenter
    data = request.json
    video_segmenter = VideoSegmentation(model_cfg, checkpoint, video_dir,global_bbox)
    set_status("Starting mask propagation through video.\nThis may take a moment...")
    update_status()
    video_segmenter.init_inference_state()
    # Check if the data contains at least one object
    if not data or len(data) == 0:
        return jsonify({'status': 'error', 'message': 'No objects provided'}), 400

    # Track the current frame index and load the image only once per frame
    current_frame_idx = None

    # Step 1: Add Points (must be done before propagation)
    for obj in data:
        ann_obj_id = obj['id']  # Object ID
        points_data = obj['points']  # Points list
        labels_data = obj['labels']  # Labels list
        frames = obj['frames']  # Frame name(s)
        ann_frame_idx = int(frames[0].split('.')[0])
        color = obj.get('borderColor')  #Color list
        
        # Store the object ID and color globally
        global_objects.append({
            'id': ann_obj_id,
            'color': color
        })
        
        # Load the frame only if it's not already loaded
        if current_frame_idx != ann_frame_idx:
            current_frame_idx = ann_frame_idx

        # Convert points and labels to NumPy arrays
        points = np.array([[int(p['x']), int(p['y'])] for p in points_data], dtype=np.float32)
        labels = np.array(labels_data, dtype=np.int32)

        # Call the add_points method for the current object
        video_segmenter.add_points(
            ann_frame_idx=ann_frame_idx,
            ann_obj_id=ann_obj_id,
            points=points,
            labels=labels
        )
    if global_bbox is not None:
        video_segmenter.del_objects_from_frame(ann_frame_idx)
    # Step 2: Propagate the segmentation through the video
    set_status("Propagating through video")
    update_status()
    video_segmenter.propagate_segmentation()
    # Step 3: Render the propagated masks and return base64 images
    set_status("Rendering video with masks")
    update_status()
    video_segmenter.render_propagated_masks(data, display_video=False, video_name="segmented_video")
    global_bbox=video_segmenter.get_yolo_data()
    print("global_bbox = ", global_bbox)
    # Free GPU memory after the task is done
    del video_segmenter
    torch.cuda.empty_cache()
    # Step 3: Render the propagated masks and return base64 images
    return jsonify({
        'status': 'success',
        'message': 'Segmentation propagated and video created!',
        'video_url': url_for('static', filename=f'/home/gauva/flask_app/static/uploads/segmented_video_with_masked.mp4')
        })
    
#fetch the global objects (id and color) to send back to the frontend
@app.route('/get_objects', methods=['GET'])
def get_objects():
    print(global_objects)
    return jsonify(global_objects), 200

# Files you want to delete
files_to_delete = ['static/uploads/output.mp4']
directories_to_clean = ['static/frames', 'static/rendered_frames', 'static/uploads']
# Function to delete files
def cleanup_files():
    for file in files_to_delete:
        if os.path.exists(file):
            os.remove(file)
            print(f"Deleted file: {file}")
        else:
            print(f"File not found: {file}")
    # Function to delete all files in a directory
    for directory in directories_to_clean:
        if os.path.exists(directory):
            # Delete all files in the directory
            for filename in os.listdir(directory):
                file_path = os.path.join(directory, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)  # Delete the file
                        print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
        else:
            print(f"Directory not found: {directory}")
#Register the cleanup function to be called at program exit
atexit.register(cleanup_files)
     
@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/get_status', methods = ['GET'])
def get_status():
    #Retrieve the status message from the session
    status = session.pop('status', "Waiting")
    return jsonify({'message' : status})

# Send status to frontend
@app.route('/update_status', methods=['GET'])
def update_status():
    """
    Send the current status to the frontend.
    This route will be called repeatedly to update the wait window.
    """
    global status_message
    return jsonify({'message': status_message})

# For demonstration, a route to change the status on the server
@app.route('/set_status/<new_status>', methods=['POST'])
def set_status(new_status):
    """
    Update the status_message (for example, called internally or via another service).
    """
    global status_message
    status_message = new_status
    return jsonify({'message': f'Status updated to "{new_status}"'}), 200
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------MAIN--------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
 
if __name__ == '__main__':
    app.run(debug=False)