# module imports
import os, json, argparse, glob, pims
import pandas as pd
import numpy as np
import frame_tracker
from collections import OrderedDict, Counter
from pathlib import Path
from functools import partial
from ast import literal_eval

from PIL import Image
from mydia import Videos
from db_utils import create_connection
from prepare_input import ProcFrameCuda, ProcFrames

# utility functions
def process_frames(frames_path, size=(416, 416)):

    # Run tests
    gpu_time_0, n_frames = ProcFrames(partial(ProcFrameCuda, size=size), frames_path)
    print(f"Processing performance: {n_frames} frames, {gpu_time_0:.2f} ms/frame")


def split_frames(data_path, perc_test):

    dataset_path = Path(data_path)
    images_path = Path(dataset_path, "images")

    # Create and/or truncate train.txt and test.txt
    file_train = open(Path(data_path, "train.txt"), "w")
    file_test = open(Path(data_path, "test.txt"), "w")

    # Populate train.txt and test.txt
    counter = 1
    index_test = int(perc_test / 100 * len(os.listdir(images_path)))
    for pathAndFilename in glob.iglob(os.path.join(images_path, "*.jpg")):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))

        if counter == index_test + 1:
            counter = 1
            file_test.write(pathAndFilename + "\n")
        else:
            file_train.write(pathAndFilename + "\n")
            counter = counter + 1
    print("Training and test set completed")


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path",
        "-o",
        help="output to txt files in YOLO format",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_format",
        "-f",
        help="format of annotations for export",
        type=str,
        default=r"yolo",
    )
    parser.add_argument(
        "--class_list",
        "-c",
        help="list of classes to use for dataset",
        type=list,
        default=[],
    )
    parser.add_argument(
        "-db",
        "--db_path",
        type=str,
        help="the absolute path to the database file",
        default=r"koster_lab.db",
        required=True,
    )
    args = parser.parse_args()

    conn = create_connection(args.db_path)

    if len(args.class_list) > 0:
        species_ref = pd.read_sql_query(
            f"SELECT id FROM species WHERE label IN {args.class_list}", conn
        )["id"].tolist()
    else:
        species_ref = pd.read_sql_query(f"SELECT id FROM species", conn)["id"].tolist()

    train_rows = pd.read_sql_query(
        f"SELECT b.filename, b.frame_number, a.species_id, a.x_position, a.y_position, a.width, a.height FROM \
        agg_annotations_frame AS a WHERE species_id IN {species_ref} LEFT JOIN subjects AS b ON a.subject_id=b.id",
        conn,
    )

    # Add dataset metadata to dataset table in koster db

    bboxes = {}
    tboxes = {}
    new_rows = []

    video_dict = {i: pims.Video(i) for i in df["filename"].unique().tolist()}

    for name, group in train_rows.groupby(["filename", "class_name", "start_frame"]):

        filename, class_name, start_frame = name

        # Track intermediate frames
        bboxes[name], tboxes[name] = [], []
        bboxes[name].extend(tuple(i) for i in group)
        tboxes[name].extend(
            frame_tracker.track_objects(
                video_dict[name[0]], class_name, bboxes[name], start_frame, 250
            )
        )

        for box in bboxes[name]:
            new_rows.append((filename, class_name, start_frame,) + box)
        for box in tboxes[name]:
            new_rows.append((filename, class_name, start_frame + box[0],) + box[1:])

    # Export txt files
    full_rows = pd.DataFrame(
        new_rows, columns=["filename", "class_name", "frame", "x", "y", "w", "h"]
    )

    txt_rows = full_rows.groupby(["filename", "frame"])

    for name, groups in txt_rows:

        file, ext = os.path.splitext(name[0])
        file_base = os.path.basename(file)

        if not os.path.isdir(args.out_path):
            os.mkdir(args.out_path)
            os.mkdir(Path(args.out_path, "images"))
            os.mkdir(Path(args.out_path, "labels"))

        if args.out_format == "yolo":
            open(f"{args.out_path}/labels/{file_base}_frame_{name[1]}.txt", "w").write(
                "\n".join(
                    [
                        "{:s} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            i[1],
                            (i[3] + i[5] / 2) / 416,
                            (i[4] + i[6] / 2) / 416,
                            i[5] / 416,
                            i[6] / 416,
                        )
                        for i in groups.values
                    ]
                )
            )

        # Save frames to image files
        Image.fromarray(video_dict[name[0]][name[1], ...]).save(
            f"{args.out_path}/images/{file_base}_frame_{name[1]}.jpg"
        )

    print("Frames extracted successfully")

    # Clear images
    gpu_res = []
    process_frames(args.out_path, size=(416, 416))

    # Create training/test sets
    split_frames(args.out_path, 0.2)


if __name__ == "__main__":
    main()
