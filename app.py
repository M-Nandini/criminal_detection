from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
from PIL import Image
import shutil
import pandas
import cv2
import os

def extract_frames(video_path, output_folder, interval=1):
    # Open the video file
    print("keeri")
    video_capture = cv2.VideoCapture(video_path)
    # Check if the video file was opened successfully
    if not video_capture.isOpened():
        print("Error: Couldn't open the video file.")
        return
    # Initialize variables
    frame_count = 0
    # Loop through the video frames
    while True:
        # Read the next frame
        ret, frame = video_capture.read()
        # If no frame is retrieved, break the loop
        if not ret:
            break
        # Save the frame if the frame count is a multiple of the interval
        if frame_count % interval == 0:
            # Save the frame as an image
            cv2.imwrite(os.path.join(output_folder, f"frame_{frame_count}.jpg"), frame)
        # Increment frame count
        frame_count += 1
    # Release the video capture object
    video_capture.release()
def get_image_paths(directory):
    image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
    image_paths = []

    # Walk through all files and directories in the given directory
    for root, dirs, files in os.walk(directory):
        for file in files:
            # Check if the file has a valid image extension
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))

    return image_paths

def process(image_paths):

    op=0
    for i in image_paths:
        #input_image="/content/output_folder/frame_129.jpg"
        input_image=i
        print(input_image)
        model = YOLO(r'.\best.pt')
        results: Results = model.predict(input_image)[0]
        #image = Image.open("/content/output_folder/frame_129.jpg")
        image = Image.open(i)
        detected_objects = []

        if hasattr(results, 'boxes') and hasattr(results, 'names'):
            for box in results.boxes.xyxy:
                object_id = int(box[-1])
                object_name = results.names.get(object_id)
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])

                detected_objects.append((object_name, (x1, y1, x2, y2)))

        # Create or clear the 'faces' directory
        if os.path.exists("faces"):
            #shutil.rmtree("faces")#NJAN CHYTHE COMMENT
            pass
            
        else:
            os.makedirs("faces")
        extracted_names = []
        # Crop and save each detected object
        for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):

            object_image = image.crop((x1, y1, x2, y2))
            object_image.save(f"faces/face{i}.jpg")
            cropped_objects_dir = "./faces/"
            for filename in os.listdir(cropped_objects_dir):
                if filename.endswith(".jpg"):
                    print(f"file name is {filename}")
                    img_path = os.path.join(cropped_objects_dir, filename)
                    model = DeepFace.find(img_path=img_path, db_path=r".\database", enforce_detection=False, model_name="Facenet512")
                    print(model)
                    if model and len(model[0]['identity']) > 0:
                        print("len inte akath keeri")
                        print(model[0]["distance"])
                        op=1
                        # Extract the name and append it to the list
                        print("model is")
                        print(model[0]["identity"],len(model[0]["identity"]))
                        
                        path=model[0]['identity'][0].split('\\')
                        
                        print(path)
                        name = path[2]
                        print(f"name is {name}")
                        # Save the known face into the 'known' folder
                        known_faces_dir=r"./known"
                        known_faces_path = os.path.join(known_faces_dir, f"{len(extracted_names) + 1}_{name}.jpg")
                        print(f"image path is {img_path}, and knownface path is {known_faces_path}")
                        shutil.copy(img_path, known_faces_path)
                        shutil.copy(input_image, known_faces_dir)
                        break#ipo latest aay add akiye

        if op==1:
            print("kitti mwoneeeee")
            shutil.copy(input_image, known_faces_dir)
           
            return input_image,img_path,True
    return input_image,img_path,False 
def clear_folder(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)
    print(files)
    # Iterate through each file and remove it
    for file_name in files:
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
def live_process(image_path):
    op=0
    print(image_path)
    
    input_image=image_path
    model = YOLO(r'./best.pt')
    results: Results = model.predict(input_image)[0]
    image = Image.open(input_image)
    detected_objects = []

    if hasattr(results, 'boxes') and hasattr(results, 'names'):
        for box in results.boxes.xyxy:
            object_id = int(box[-1])
            object_name = results.names.get(object_id)
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            print(x1,y1,x2,y2)
            detected_objects.append((object_name, (x1, y1, x2, y2)))

    # Create or clear the 'faces' directory
    print("faces ondakan pon")
    if os.path.exists("faces"):
        pass
        #shutil.rmtree("faces")#NJAN CHYTHE COMMENT
    else:
        os.makedirs("faces")
    extracted_names = []
    # Crop and save each detected object
    for i, (object_name, (x1, y1, x2, y2)) in enumerate(detected_objects):

        object_image = image.crop((x1, y1, x2, y2))
        object_image.save(f"faces/face{i}.jpg")
        cropped_objects_dir = "./faces/"
        for filename in os.listdir(cropped_objects_dir):
            if filename.endswith(".jpg"):
                print(f"file name is {filename}")
                img_path = os.path.join(cropped_objects_dir, filename)
                #img_path="/content/faces/face0.jpg"
                model = DeepFace.find(img_path=img_path, db_path=r".\database", enforce_detection=False, model_name="VGG-Face")
                if model and len(model[0]['identity']) > 0:
                    op=1
                    # Extract the name and append it to the list
                    print("model is")
                    print(model[0]["identity"],len(model[0]["identity"]))
                    
                    path=model[0]['identity'][0].split('\\')
                    
                    print(path)
                    name = path[2]#play with the indexz bae
                    print(f"name is {name}")
                    # Save the known face into the 'known' folder
                    known_faces_dir=r"./known"
                    known_faces_path = os.path.join(known_faces_dir, f"{len(extracted_names) + 1}_{name}.jpg")
                    print(f"image path is {img_path}, and knownface path is {known_faces_path}")
                    shutil.copy(img_path, known_faces_path)
                    shutil.copy(input_image, known_faces_dir)

    if op==1:
        print("kitti mwoneeeee")
        
        shutil.copy(input_image, known_faces_dir)
        return True
        
        #cv2.imshow('Video', i)
        #frame edkane koode add akanmcode
        return input_image
        