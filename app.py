# -*- coding: utf-8 -*-
import streamlit as st
import altair as alt
import pandas as pd
import numpy as np
import os, urllib, cv2, pims, json
import sqlite3
import db_utils

from fast import KosterModel
from collections import OrderedDict

# Disable automatic encoding warning for uploaded files
st.set_option("deprecation.showfileUploaderEncoding", False)


def main():
    # Set up appearance of sidebar
    st.sidebar.title("Koster Lab Live Demo")
    st.sidebar.image(
        "https://panoptes-uploads.zooniverse.org/production/project_avatar/86c23ca7-bbaa-4e84-8d8a-876819551431.png",
        use_column_width=True,
    )
    # Run main app
    run_the_app()


@st.cache(allow_output_mutation=True)
def load_network():
    m = KosterModel()
    return m


def run_the_app():

    @st.cache
    def load_data():
        db_path = "/data/database/demo.db"
        movie_dir = "/uploads"
        conn = db_utils.create_connection(db_path)

        df = pd.read_sql_query(
            "SELECT b.filename, b.frame_number, a.species_id, a.x_position, a.y_position, a.width, a.height FROM agg_annotations_frame AS a LEFT JOIN subjects AS b ON a.subject_id=b.id",
            conn,
        )

        df["movie_path"] = (
            movie_dir
            + "/"
            + df["filename"].apply(
                lambda x: os.path.basename(x.rsplit("_frame_")[0]) + ".mov"
            )
        )

        return df

    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()
    m = load_network()

    # Default is to load images
    if st.sidebar.checkbox("Custom File Upload", value=True):

        img_file_buffer = st.file_uploader(
            "Upload an image/video (maximum size 200MB). Supported formats: png, jpg, jpeg, mov, mp4",
            type=["png", "jpg", "jpeg", "mov", "mp4"],
        )

        if img_file_buffer is not None:

            # text_io = io.TextIOWrapper(img_file_buffer)
            raw_buffer = img_file_buffer.read()
            bytes_as_np_array = np.fromstring(raw_buffer, np.uint8)

            # if image
            try:
                image = cv2.imdecode(bytes_as_np_array, -1)
                # Resize the image to the size YOLO model expects
                selected_frame = cv2.resize(image, (416, 416))
                # Save in a temp file as YOLO expects filepath
                cv2.imwrite("/data/predicted_image.jpg", selected_frame)
                selected_frame = "/data/predicted_image.jpg"
            # if video
            except:
                video = True
                with open("/data/predicted_video.mp4", "wb") as out_file:  # open for [w]riting as [b]inary
                     out_file.write(raw_buffer)

                selected_frame = "/data/predicted_video.mp4"

        else:

            # Show the last image
            selected_frame = "/data/predicted_image.jpg"

    else:
        
        df = load_data()
        # Load all movies to speed up frame retrieval
        movie_dict = OrderedDict({i: pims.Video(i) for i in df["movie_path"].unique()})

        files = df["filename"].tolist()
        
        selected_frame_index, selected_frame = frame_selector_ui(df, movie_dict, files)

        if selected_frame_index == None:
            st.error(
                "No frames fit the criteria. Please select different label or number."
            )
            return

    # Load the image from S3.
    m.source = selected_frame
    m.conf_thres = confidence_threshold
    m.iou_thres = overlap_threshold

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    processed_image, vid = m.detect()
    if vid:
        st.header("Real-time Computer Vision")
        st.markdown("**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
                    % (overlap_threshold, confidence_threshold))
        st.video("/data/testapi/predicted_video.mp4")
    else:
        draw_image_with_boxes(
            processed_image,
            "Real-time Computer Vision",
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold),
            )

@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(df, selected_movie_path):
    return df[df.movie_path == selected_movie_path]

# This sidebar UI is a little search engine to find certain object types.
def frame_selector_ui(df, movie_dict, files):

    st.sidebar.markdown("# Movie")

    # Choose a movie out of the selected movies.
    selected_movie_index = st.sidebar.slider(
        "Choose a movie (index)", 0, len(movie_dict), 0
    )

    selected_movie_path, selected_movie = list(movie_dict.items())[selected_movie_index]

    st.sidebar.markdown("# Frame")

    # Choose a frame out of the selected frames.
    selected_frame_index = st.sidebar.slider(
        "Choose a frame (index)", 0, len(get_selected_frames(df, selected_movie_path)) - 1, 0
    )

    selected_frame = selected_movie[selected_frame_index]
    # Resize the image to the size YOLO model expects
    selected_frame = cv2.resize(selected_frame, (416, 416))
    # Save in a temp file as YOLO expects filepath
    cv2.imwrite("/data/predicted_image.jpg", selected_frame)
    return selected_frame_index, "/data/predicted_image.jpg"


# This sidebar UI lets the user select parameters for the YOLO object detector.
def object_detector_ui():
    st.sidebar.markdown("# Model")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(image_with_boxes, header, description):
    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(image_with_boxes.astype(np.uint8), use_column_width=True)


if __name__ == "__main__":
    main()
                                  