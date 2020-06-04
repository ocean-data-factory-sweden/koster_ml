# module imports
import os, json, argparse, glob, pims
import pandas as pd
import numpy as np
import frame_tracker
from collections import OrderedDict, Counter
from pathlib import Path
from functools import partial
from ast import literal_eval
from tqdm import tqdm

from PIL import Image
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
    parser.add_argument(
        "-m",
        "--movie_dir",
        type=str,
        help="the directory of movie files",
        default=r"/uploads/",
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
        agg_annotations_frame AS a LEFT JOIN subjects AS b ON a.subject_id=b.id WHERE species_id IN {tuple(species_ref)}",
        conn,
    )

    # Add dataset metadata to dataset table in koster db

    bboxes = {}
    tboxes = {}
    new_rows = []

    train_rows["movie_path"] = (
        args.movie_dir
        + "/"
        + train_rows["filename"].apply(
            lambda x: os.path.basename(x.rsplit("_frame_")[0]) + ".mov"
        )
    )

    video_dict = {i: pims.Video(i) for i in train_rows["movie_path"].unique()}

    train_rows = train_rows[
        [
            "filename",
            "species_id",
            "frame_number",
            "movie_path",
            "x_position",
            "y_position",
            "width",
            "height",
        ]
    ]

    for name, group in tqdm(
        train_rows.groupby(["filename", "species_id", "frame_number", "movie_path"])
    ):

        filename, species_id, frame_number, movie_path = name[:4]

        named_tuple = tuple([filename, species_id, frame_number, movie_path])

        # Track intermediate frames
        bboxes[named_tuple], tboxes[named_tuple] = [], []
        bboxes[named_tuple].extend(tuple(i[4:]) for i in group.values)
        tboxes[named_tuple].extend(
            frame_tracker.track_objects(
                video_dict[name[3]],
                species_id,
                bboxes[named_tuple],
                frame_number,
                frame_number + 10,
            )
        )

        for box in bboxes[named_tuple]:
            new_rows.append(
                (
                    filename,
                    species_id,
                    frame_number,
                    movie_path,
                    video_dict[name[3]][frame_number].shape[1],
                    video_dict[name[3]][frame_number].shape[0],
                )
                + box
            )
        for box in tboxes[named_tuple]:
            new_rows.append(
                (
                    filename,
                    species_id,
                    frame_number + box[0],
                    movie_path,
                    video_dict[name[3]][frame_number].shape[1],
                    video_dict[name[3]][frame_number].shape[0],
                )
                + box[1:]
            )

    # Export txt files
    full_rows = pd.DataFrame(
        new_rows,
        columns=[
            "filename",
            "species_id",
            "frame",
            "movie_path",
            "f_w",
            "f_h",
            "x",
            "y",
            "w",
            "h",
        ],
    )

    for name, groups in full_rows.groupby(["filename", "frame", "movie_path"]):

        file, ext = os.path.splitext(name[2])
        file_base = os.path.basename(file)

        if not os.path.isdir(args.out_path):
            os.mkdir(Path(args.out_path, "images"))
            os.mkdir(Path(args.out_path, "labels"))

        if args.out_format == "yolo":
            open(f"{args.out_path}/labels/{file_base}_frame_{name[1]}.txt", "w").write(
                "\n".join(
                    [
                        "{} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            i[1],
                            (i[6] + i[8] / 2) / i[4],
                            (i[7] + i[9] / 2) / i[5],
                            i[8] / i[4],
                            i[9] / i[5],
                        )
                        for i in groups.values
                    ]
                )
            )

        # Save frames to image files
        Image.fromarray(video_dict[name[2]][name[1]]).save(
            f"{args.out_path}/images/{file_base}_frame_{name[1]}.jpg"
        )

    print("Frames extracted successfully")

    # Clear images
    # gpu_res = []
    process_frames(args.out_path + "/images", size=(416, 416))

    # Create training/test sets
    split_frames(args.out_path, 0.2)


if __name__ == "__main__":
    main()
