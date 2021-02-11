# -*- coding: utf-8 -*-
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2, pims, json, random
import sqlite3
import db_utils

from fast import KosterModel
from collections import OrderedDict

# Disable automatic encoding warning for uploaded files
st.set_option("deprecation.showfileUploaderEncoding", False)

# interact with FastAPI endpoint
backend = "http://fastapi:8000/"

def main():
    # Set up appearance of sidebar
    st.sidebar.title("Koster Lab - Deep Sea Coral Detection")
    st.sidebar.image(
        "https://panoptes-uploads.zooniverse.org/production/project_avatar/86c23ca7-bbaa-4e84-8d8a-876819551431.png",
        use_column_width=True,
    )
    # Run main app
    run_the_app()
    st.sidebar.image(
        "https://panoptes-uploads.zooniverse.org/production/project_attached_image/99429003-51ae-4667-b9b0-7ec2ff518723.png",
        use_column_width=True,
    )


@st.cache(allow_output_mutation=True)
def predict(media_path: str, conf_thres: float, iou_thres: float, endpoint: str=backend+'/predict'):
    r = requests.post(
        endpoint, data={"media_path": media_path, "conf_thres": conf_thres, 
                          "iou_thres": iou_thres}, timeout=8000
    )
    return r

@st.cache
def load_data(endpoint=backend+'/data'):
    r = requests.get(
        endpoint, data={}, timeout=8000
    )
    return r

@st.cache
def get_movie_frame(file_path: str, frame_number: int, endpoint: str=backend+'/read'):
    r = requests.get(
        endpoint, data={"file_path": file_path, "frame_number": frame_number}, timeout=8000
    )
    return r["frame_data"]


def run_the_app():
    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()
    # st.markdown(
    #    "Instructions: Use the sliders to adjust the model hyperparameters and wait to see the impact on the predicted bounding boxes."
    # )

    # Default is to load images
    if st.sidebar.checkbox("Custom File Upload", value=True):

        st.warning(
            "Disclaimer: By uploading your files here, you also accept that any uploaded files will be processed on an external server located within the EU. \
            You also accept that these files may be stored and used for training purposes in future model iterations. At your request, any data provided will be removed from our servers \
            in accordance with prevailing GDPR regulations."
        )

        img_file_buffer = st.file_uploader(
            "Upload an image/video (maximum size 200MB). Supported formats: png, jpg, jpeg, mov, mp4. Instructions: Use the sliders to adjust the model hyperparameters and wait to see the impact on the predicted bounding boxes.",
            type=["png", "jpg", "jpeg", "mov", "mp4"],
        )

        if img_file_buffer is not None:
            name = img_file_buffer.name
            # text_io = io.TextIOWrapper(img_file_buffer)
            raw_buffer = img_file_buffer.read()
            bytes_as_np_array = np.fromstring(raw_buffer, np.uint8)
            # if image
            try:
                image = cv2.imdecode(bytes_as_np_array, -1)
                # Resize the image to the size YOLO model expects
                selected_frame = image  # cv2.resize(image, (416, 416))

                # Save in a temp file as YOLO expects filepath
                cv2.imwrite(f"{m.out}/{name}", selected_frame)
                selected_frame = f"{m.out}/{name}"
            # if video
            except:
                video = True

                with open(
                    f"{os.path.dirname(m.out)}/{name}", "wb"
                ) as out_file:  # open for [w]riting as [b]inary
                    out_file.write(raw_buffer)

                vid_out_path = f"{m.out}/{name}"
                selected_frame = f"{os.path.dirname(m.out)}/{name}"

        else:
            # Show the last image
            st.error("No file uploaded. Please select a file from your computer.")
            return

    else:
        if not os.path.exists("/data/api/training_footage"):
            os.mkdir("/data/api/training_footage")  # create dest dir
        
        m.out = "/data/api/training_footage"
        # Load classified data
        df = load_data()
        # Load all movies to speed up frame retrieval
        movie_list = [i for i in df["movie_path"].unique()]

        # Select a movie
        selected_movie_path = movie_selector_ui(movie_list)
        movie_frames = get_selected_frames(df, selected_movie_path)

        # Select frame
        selected_frame_index = frame_selector_ui(movie_frames)
        selected_frame_number = movie_frames.iloc[selected_frame_index]
        selected_frame = get_movie_frame(selected_movie_path, selected_frame_number)

        # Resize the image to the size YOLO model expects
        # selected_frame = cv2.resize(selected_frame, (416, 416))
        # Convert color space to match YOLO input
        selected_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
        # Save in a temp file as YOLO expects filepath
        mbase = os.path.basename(selected_movie_path).split('.')[0]
        cv2.imwrite(f"{m.out}/{mbase}_{selected_frame_number}.png", selected_frame)
        selected_frame = f"{m.out}/{mbase}_{selected_frame_number}.png"

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    processed_image, vid = predict(media_path=selected_frame, conf_thres=confidence_threshold, iou_thres=overlap_threshold)
    if vid:
        st.header("Model Output")
        st.markdown(
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold)
        )
        st.video(vid_out_path)
        os.remove(selected_frame)
    else:
        # Draw the header and image.
        st.subheader("Model Output")
        st.markdown("**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold))
        st.image(selected_frame, use_column_width=True)
        os.remove(selected_frame)


@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(df, selected_movie_path):
    return df[df.movie_path == selected_movie_path]["frame_number"]


# This sidebar UI is a little search engine to find certain object types.
def movie_selector_ui(movie_dict):

    st.sidebar.markdown("# Movie")

    # Choose a movie out of the selected movies.
    selected_movie_index = st.sidebar.slider(
        "Choose a movie (index)", 0, len(movie_list) - 1, 0
    )

    selected_movie_path = movie_list[selected_movie_index]
    st.sidebar.markdown(f"Selected movie: {os.path.basename(selected_movie_path)}")

    return selected_movie_path


# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(movie_frames):

    st.sidebar.markdown("# Frame")

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider(
        "Choose a frame (index)", 0, len(movie_frames) - 1, 0
    )

    return selected_frame_index


# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.subheader("Model hyperparameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold

if __name__ == "__main__":
    main()
