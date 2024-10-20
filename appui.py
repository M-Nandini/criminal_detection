import streamlit as st
import cv2  # Assuming you have OpenCV installed
import numpy as np
from app import extract_frames,process, get_image_paths,live_process,clear_folder  # Import backend functions
from ultralytics import YOLO
from ultralytics.engine.results import Results
from deepface import DeepFace
from PIL import Image
from streamlit_folium import st_folium
from streamlit_folium import folium_static
import shutil
import pandas
import cv2
import os
import time
import folium
import re
def main():
    """Streamlit app for criminal detection in video"""

    st.title("Surviellance App")
    st.write("Upload a video file or use your livecam.")

    
    source_option = st.selectbox("Source", ["Video Upload", "Webcam"])

    if source_option == "Video Upload":
        uploaded_files = st.file_uploader("Choose a video...", type=["mp4", "avi"], accept_multiple_files=True)
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                st.video(uploaded_file)
                filename = uploaded_file.name
                video_bytes = uploaded_file.read()
                video_folder = "videos"
                os.makedirs(video_folder, exist_ok=True)
                with open(os.path.join(video_folder, filename), "wb") as f:
                    f.write(video_bytes)

            #video_bytes = uploaded_files.read()
            #filename = uploaded_files.name
   
    else:
        video_capture = cv2.VideoCapture(0)  

    
    database = st.text_input("Image Database Path")
    if not database:
        st.error("Please provide a image database path.")
        return

    
    if st.button("Start Detection"):
        output_folder = "output_folder"
        video_folder = "videos"
        try:
            os.makedirs(video_folder, exist_ok=True)  
        except OSError as e:
            st.error(f"Error creating folder: {e}")
            return
        try:
            os.makedirs(output_folder, exist_ok=True)

           
            if source_option == "Video Upload":
                
                for video_file in os.listdir(video_folder):
                    video_path = os.path.join(video_folder, video_file)
                    st.write(f"Processing video: {video_path}")
                    extract_frames(video_path, output_folder)
                    image_paths = get_image_paths(output_folder)
                    image,img,a = process(image_paths)
                    print(a)
                    if a == True:
                        print(f"video path is {video_path}")
                        st.header("Person Found")
                        #ime = Image.open(image)
                        col1, col2 ,col3= st.columns(3)
                        with col1:
                            st.image(image,width=400)
                        with col2:
                            st.image(img)
                        match = re.search(r'\((-?\d+\.\d+),\s*(-?\d+\.\d+)\)\.mp4', video_path)
                        print(f"match is {match}")
                        if match:
                            lat = float(match.group(1))
                            lon = float(match.group(2))
                            print(lat,lon)
                            
                            #st.map(latitude=8.511018,longitude=76.958195)
                            my_map = folium.Map(location=(lat,lon), zoom_start=30)
                            folium.Marker(location=[lat, lon], popup="Your Location").add_to(my_map)
                            with col3:
                                folium_static(my_map)
                            #st_folium(my_map,width=600)
                            break
                        #image = Image.open(image)
                        #st.image(image.resize(600,400))
                        #break #venel break eyam ipo or ale kitmbo nirthanel
                    else:
                        st.write("No face found in the given video.")
                        clear_folder("output_folder")
                clear_folder("videos")
                clear_folder("faces")
            else:
                t=0
                while True:
                    t=t+1
                    ret, frame = video_capture.read()
                    if not ret:
                        break
                    path=os.path.join(output_folder, f"frame_{t}.jpg")
                    cv2.imwrite(path, frame)
                    V=live_process(path)
                    if V==True:
                        st.header("Person Found")
                        col_1,col_2=st.columns(2)
                        with col_1:
                            st.image(path, f"frame_{t}.jpg")
                        lat =  8.4706245 
                        lon = 76.9790898
                        print(lat,lon)
                        
                        #st.map(latitude=8.511018,longitude=76.958195)
                        my_map = folium.Map(location=(lat,lon), zoom_start=30)
                        folium.Marker(location=[lat, lon], popup="Your Location").add_to(my_map)
                        with col_2:
                            folium_static(my_map)
                        break
                    #cv2.imshow('Video', frame)#ith  veno

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            
            clear_folder("videos")
            clear_folder("faces")
    

        except Exception as e:
            st.error(f"An error occurred: {e}")
        #finally:
           
            #shutil.rmtree(output_folder, ignore_errors=True) 

  
    if source_option == "Webcam":
        video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

