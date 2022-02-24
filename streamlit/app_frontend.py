# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os, requests, cv2, json
import pims
import base64, zlib
from PIL import Image

# Set app config
st.set_page_config(
    page_title="Koster Seafloor Detector App", page_icon="assets/favicon-16x16.png"
)

# Fix style issues
hide_streamlit_style = """
            <style>
.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
text-align: center;
}
footer {
    visibility: hidden;
    }
.css-1l02zno {
    background-attachment: fixed;
    box-sizing: border-box;
    flex-shrink: 0;
    height: 100vh;
    overflow: auto;
    padding: 0rem 1rem;
    position: relative;
    transition: margin-left .3s,box-shadow .3s;
    width: 21rem;
    z-index: 100;

}
</style>
<div class="footer">
<p>Powered by <a style='display: block; text-align: center;' href="https://www.combine.se/" target="_blank">Combine</a></p>
</div>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Disable automatic encoding warning for uploaded files
st.set_option("deprecation.showfileUploaderEncoding", False)

# interact with FastAPI endpoint
backend = "http://fastapi:5000"

def main():
    # Set up appearance of sidebar
    st.sidebar.header("Koster Seafloor Detector")
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
def predict(
    media_path: str,
    conf_thres: float,
    iou_thres: float,
    endpoint: str = backend + "/predict",
):
    r = requests.post(
        endpoint,
        params={
            "media_path": media_path,
            "conf_thres": conf_thres,
            "iou_thres": iou_thres,
        },
        timeout=8000,
    )
    return (
        r.json()["prediction"],
        r.json()["vid"],
        r.json()["prediction_dict"],
    )


@st.cache
def load_data(endpoint=backend + "/data"):
    r = requests.get(endpoint, params={}, timeout=8000)
    return pd.DataFrame.from_dict(r.json()["data"])


@st.cache(allow_output_mutation=True)
def get_movie_frame(
    file_path: str, frame_number: int, endpoint: str = backend + "/read"
):
    r = requests.get(
        endpoint,
        params={"file_path": file_path, "frame_number": frame_number},
        timeout=8000,
    )
    if r.ok:
        return np.array(json.loads(r.json()["frame_data"]))
    else:
        return None


@st.cache
def save_image(file_name: str, file_data, endpoint: str = backend + "/save"):
    r = requests.post(
        endpoint,
        params={"file_name": file_name},
        files={"file_data": file_data},
        timeout=8000,
    )
    return r.json()["output"]


@st.cache
def save_video(
    file_name: str,
    file_data,
    fps: int,
    w: int,
    h: int,
    endpoint: str = backend + "/save_vid",
):
    r = requests.post(
        endpoint,
        params={"file_name": file_name, "fps": fps, "w": w, "h": h},
        files={"file_data": zlib.compress(file_data)},
        timeout=8000,
    )
    return r.json()["output"]


def unswedify(string):
    """Convert ä and ö to utf-8"""
    return (
        string.encode("utf-8")
        .replace(b"\xc3\xa4", b"a\xcc\x88")
        .replace(b"\xc3\xb6", b"a\xcc\x88")
        .decode("utf-8")
    )


def get_table_download_link(json_dict):
    """Generates a link allowing the data in a given panda dataframe to be downloaded
    in:  dataframe
    out: href string
    """
    dlist = [
        [key, i[0], i[1], i[2], i[3]] for key, value in json_dict.items() for i in value
    ]
    df = pd.DataFrame.from_records(
        dlist, columns=["filename", "frame_no", "annotation", "class_id", "conf"]
    )
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(
        csv.encode()
    ).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="annotations.csv">Download annotations file</a>'
    return href


def run_the_app():
    # Draw the UI element to select parameters for the YOLO object detector.
    confidence_threshold, overlap_threshold = object_detector_ui()
    st.header("Welcome to our underwater *world!* :fish:")
    st.subheader("Here is a quick FAQ to get you started.")

    with st.beta_expander("What is the Koster Seafloor Detector?"):
            st.info(
                "Originally developed in collaboration between ODF, SEAnalytics, Wildlife.ai and the University of Gothenburg, the Koster Seafloor Detector is an object detection tool for different species in subsea images/videos. \
                The model is based on an open-source implementation of a popular single-shot object detection architecture, YOLO. The model is trained on footage from the protected Koster Marine Park on the west coast of Sweden. "
            )
            st.image("https://panoptes-uploads.zooniverse.org/production/project_attached_image/e4794aff-ebb5-4a59-9b8a-91c52bc8dede.jpeg")
    with st.beta_expander("How do I use this?"):
            st.info(
                "The easiest way is to look through footage that has already been classified. Start by clicking on the tickbox in the side bar called 'Custom File Upload' and you should see an \
                image with a bounding box containing a name and confidence level. There you should now see sidebar options to filter by movie and movie frame. \
                In the sidebar, you can now adjust the confidence and overlap thresholds to see how this impacts the output of the model. \
                Once you get the hang of it, start uploading your own custom footage by clicking on the tickbox again. Click on 'Browse files' and add an image/video from your own computer and wait to see the model output." 
            )
    with st.beta_expander("Which species are currently supported?"):
            st.info(
                "In this version, we currently support four marine classes. These are deep sea corals, deeplet sea anemones, football sponges and fish (general). This is continuously updated and the performance across these \
                groups may not be equal due to the differences in the available volume of footage for each class."
            )
    with st.beta_expander("I don't see any bounding boxes anywhere, what can I do?"):
            st.info(
                "There are multiple possible reasons for a lack of bounding boxes. For example, the environment in your footage may be too different from what our model expects. However, \
                a good starting point is to decrease the confidence threshold to see if this results in the appearance of bounding boxes. "
            )
    with st.beta_expander("What happens to my footage when I upload it here? Can I get it removed?"):
            st.info(
                "By uploading your files here, you also accept that any uploaded files will be processed on an external server located within the EU. \
            You also accept that these files may be stored and used for training purposes in future model iterations. At your request, any data provided will be removed from our servers \
            in accordance with prevailing GDPR regulations."
            )
    with st.beta_expander("I want the app to be even better, is there a way I can contribute to the model training?"):
            st.info(
                "Thank-you for your interest in improving our model. We are of course open to any questions, feedback or suggestions, which you may forward to jurie.germishuys@combine.se. You can also contribute by annotating footage for \
                future iterations on our citizen science project page https://www.zooniverse.org/projects/victorav/the-koster-seafloor-observatory."
            )
    
     

    # Default is to load images
    if st.sidebar.checkbox("Custom File Upload", value=True):
        custom = True
        st.empty()
        st.subheader("Upload footage")
        img_file_buffer = st.file_uploader(
            "Upload an image/video (maximum size 1GB). Supported formats: png, jpg, jpeg, mov, mp4.",
            type=["png", "jpg", "jpeg", "mov", "mp4"],
        )

        if img_file_buffer is not None:
            name = img_file_buffer.name
            im = os.path.splitext(name)[1].lower() in [".png", ".jpg", ".jpeg"]
            # text_io = io.TextIOWrapper(img_file_buffer)
            raw_buffer = img_file_buffer.read()

            if im:
                try:
                    # image = cv2.imdecode(np.fromstring(raw_buffer, np.uint8), -1)
                    # Resize the image to the size YOLO model expects
                    # selected_frame = image  # cv2.resize(image, (416, 416))
                    # selected_frame = np.float32(image)
                    # selected_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
                    # Save in a temp file as YOLO expects filepath
                    selected_frame = save_image(f"{name}", raw_buffer)
                except:
                    selected_frame = f"/data/api/{name}"

            else:
                video = True
                try:
                    with open(
                        f"temp_{name}", "wb"
                    ) as out_file:  # open for [w]riting as [b]inary
                        out_file.write(raw_buffer)

                    vid_cap = cv2.VideoCapture(f"temp_{name}")
                    fps = int(vid_cap.get(cv2.CAP_PROP_FPS))
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    assert fps > 0
                    selected_frame = save_video(f"{name}", raw_buffer, fps, w, h)
                    os.remove(f"temp_{name}")
                except:
                    selected_frame = f"/data/api/{name}"

        else:
            # Show the last image
            st.error("No file uploaded. Please select a file from your computer.")
            return

    else:
        custom = False
        # st.error("This feature will allow you to explore our datasets. Please upload your own media until this becomes available. ")
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
        if selected_frame is None:
            selected_frame = get_movie_frame(
                unswedify(selected_movie_path), selected_frame_number
            )
        selected_frame = np.float32(selected_frame)
        selected_frame = cv2.cvtColor(selected_frame, cv2.COLOR_RGB2BGR)
        # selected_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
        # Save in a temp file as YOLO expects filepath
        mbase = os.path.basename(selected_movie_path).split(".")[0]
        cv2.imwrite(f"{mbase}_{selected_frame_number}.jpeg", selected_frame)
        with open(f"{mbase}_{selected_frame_number}.jpeg", "rb") as out_file:
            image_data = out_file.read()
        selected_frame = save_image(f"{mbase}_{selected_frame_number}.jpeg", image_data)
        os.remove(f"{mbase}_{selected_frame_number}.jpeg")

    # Get the boxes for the objects detected by YOLO by running the YOLO model.
    processed_image, vid, detect_dict = predict(
        media_path=selected_frame,
        conf_thres=confidence_threshold,
        iou_thres=overlap_threshold,
    )
    if vid:
        st.header("Model Output")
        st.markdown(
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold)
        )
        st.video("".join(processed_image))
        #st.video(bytes(list(processed_image)))
        st.markdown(get_table_download_link(detect_dict), unsafe_allow_html=True)
        # os.remove(selected_frame)
    else:
        # Draw the header and image.
        st.subheader("Model Output")
        st.markdown(
            "**YOLO v3 Model** (overlap `%3.1f`) (confidence `%3.1f`)"
            % (overlap_threshold, confidence_threshold)
        )
        # if not custom:
        #    st.image(processed_image, use_column_width=True)
        st.image(
            cv2.cvtColor(np.float32(processed_image) / 255, cv2.COLOR_BGR2RGB),
            use_column_width=True,
        )
        st.markdown(get_table_download_link(detect_dict), unsafe_allow_html=True)
        # os.remove(selected_frame)


@st.cache(hash_funcs={np.ufunc: str})
def get_selected_frames(df, selected_movie_path):
    return df[df.movie_path == selected_movie_path]["frame_number"]


# This sidebar UI is a little search engine to find certain object types.
def movie_selector_ui(movie_list):

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
    # st.sidebar.subheader("Model hyperparameters")
    confidence_threshold = st.sidebar.slider(
        "Confidence threshold", 0.0, 1.0, 0.5, 0.01
    )
    overlap_threshold = st.sidebar.slider("Overlap threshold", 0.0, 1.0, 0.3, 0.01)
    return confidence_threshold, overlap_threshold


if __name__ == "__main__":
    main()
