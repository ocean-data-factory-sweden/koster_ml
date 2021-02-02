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
def load_network():
    m = KosterModel()
    return m


def run_the_app():
    @st.cache
    def load_data():
        db_path = "/data/db_config/koster_lab-nm.db"
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
    m = load_network()
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
                cv2.imwrite(f"{m.dest}/{name}", selected_frame)
                selected_frame = f"{m.dest}/{name}"
            # if video
            except:
                video = True

                with open(
                    f"{m.dest}/{name}", "wb"
                ) as out_file:  # open for [w]riting as [b]inary
                    out_file.write(raw_buffer)

                selected_frame = f"{m.dest}/{name}"

        else:
            # Show the last image
            st.error("No file uploaded. Please select a file from your computer.")
            return

    else:
        # Generate temp id
        fid = random.randint(100000000000, 999999999999)
        # Load classified data
        df = load_data()
        # Load all movies to speed up frame retrieval
        movie_dict = OrderedDict({i: pims.Video(i) for i in df["movie_path"].unique()})

        # Select a movie
        selected_movie_path, selected_movie = movie_selector_ui(movie_dict)
        movie_frames = get_selected_frames(df, selected_movie_path)

        # Select frame
        selected_frame_index = frame_selector_ui(movie_frames)
        selected_frame_number = movie_frames.iloc[selected_frame_index]
        selected_frame = selected_movie[selected_frame_number]

        # Resize the image to the size YOLO model expects
        # selected_frame = cv2.resize(selected_frame, (416, 416))
        # Convert color space to match YOLO input
        selected_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
        # Save in a temp file as YOLO expects filepath
        cv2.imwrite(f"{m.out}/temp_{fid}.png", selected_frame)
        selected_frame = f"{m.out}/temp_{fid}.png"

    # Load the image from S3.
    m.source = selected_frame
    m.conf_thres = confidence_threshold
    m.iou_thres = overlap_threshold

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    processed_image, vid = m.detect()
    if vid:
        st.header("Model Output")
        st.markdown(
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold)
        )
        st.video(f"{m.out}/temp_{fid}.mp4")
        os.remove(f"{m.out}/temp_{fid}.mp4")
    else:
        draw_image_with_boxes(
            fid,
            processed_image,
            "Model Output",
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold),
        )
        os.remove(f"{m.out}/temp_{fid}.png")


@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(df, selected_movie_path):
    return df[df.movie_path == selected_movie_path]["frame_number"]


# This sidebar UI is a little search engine to find certain object types.
def movie_selector_ui(movie_dict):

    st.sidebar.markdown("# Movie")

    # Choose a movie out of the selected movies.
    selected_movie_index = st.sidebar.slider(
        "Choose a movie (index)", 0, len(movie_dict) - 1, 0
    )

    selected_movie_path, selected_movie = list(movie_dict.items())[selected_movie_index]
    st.sidebar.markdown(f"Selected movie: {os.path.basename(selected_movie_path)}")

    return selected_movie_path, selected_movie


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


# Draws an image with boxes overlayed to indicate the presence of cars, pedestrians etc.
def draw_image_with_boxes(fid, image_with_boxes, header, description):
    # Draw the header and image.
    st.subheader(header)
    st.markdown(description)
    st.image(f"{m.out}/temp_{fid}.png", use_column_width=True)


if __name__ == "__main__":
    main()
