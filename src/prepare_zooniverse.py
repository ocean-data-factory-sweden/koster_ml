# module imports

import os, json
import pandas as pd
import numpy as np
import argparse
from collections import OrderedDict
from pathlib import Path
from ast import literal_eval
import frame_tracker
from db_setup import *

# utility functions

def get_id(conn, row):

    # Currently we discard sites that have no lat or lon coordinates, since site descriptions are not unique
    # it becomes difficult to match this information otherwise

    try:
        gid = retrieve_query(conn, f"SELECT id FROM species WHERE label=='{row['class_name']}'")[0][0]
    except IndexError:
        gid = 0
    return gid

def prepare(classifications_path):
    df = pd.read_csv(classifications_path)
    df = df[["classification_id", "annotations", "subject_data"]]

    # Extract the video filename and annotation details
    df["annotation"] = df.apply(
        lambda x: (
            [v["filename"] for k, v in json.loads(x.subject_data).items()],
            literal_eval(x["annotations"])[0]["value"],
        )
        if len(literal_eval(x["annotations"])[0]["value"]) > 0
        else None,
        1,
    )

    # Convert annotation to format which the tracker expects
    ds = [
        OrderedDict(
            {
                "filename": i[0][0].split("_frame", 1)[0],
                "class_name": i[1][0]["tool_label"],
                "start_frame": int(i[0][0].split("_frame", 1)[1].replace(".jpg", "")),
                "x": int(i[1][0]["x"]),
                "y": int(i[1][0]["y"]),
                "w": int(i[1][0]["width"]),
                "h": int(i[1][0]["height"]),
            }
        )
        for i in df.annotation
        if i is not None
    ]
    return pd.DataFrame(ds)


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--w2_path",
        "-w2",
        help="output from workflow 2 in Zooniverse",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--out_path",
        "-o",
        help="output to txt files in YOLO format",
        type=str,
        required=True,
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

    train_rows = prepare(args.w2_path)

    bboxes = {}
    tboxes = {}
    new_rows = []

    for name, group in train_rows.groupby(["filename", "class_name", "start_frame"]):
        filename, class_name, start_frame = name
        # temp for testing
        filename = "../data/data_example/test_video.mp4"
        # Determine a threshold for overlap in terms of IOU to discard duplicate detections (TBD)
        bboxes[name], tboxes[name] = [], []
        bboxes[name].extend((i[3], i[4], i[5], i[6]) for i in group.values)
        tboxes[name].extend(
            frame_tracker.track_objects(
                filename, class_name, bboxes[name], start_frame, 250
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

        if not os.path.isdir(args.out_path): os.mkdir(args.out_path)

        open(f"{args.out_path}/{file_base}_frame_{name[1]}.txt", "w").write(
            "\n".join(
                ["{:s} {:.6f} {:.6f} {:.6f} {:.6f}".format(
                    i[1],
                    (i[3] + i[5] / 2) / 416,
                    (i[4] + i[6] / 2) / 416,
                    i[5] / 416,
                    i[6] / 416) for i in groups.values]
            )
        )

    # Add annotations to db

    # Get species_id from species table
    conn = create_connection(args.db_path)

    full_rows['species_id'] = full_rows.apply(lambda x: get_id(conn, x), 1)

    db_rows = full_rows.drop(columns="class_name")[["species_id", "x", "y", "w", "h", "frame"]]#.groupby(["filename", "species_id", "frame"])

    try:
        insert_many(conn, [tuple(i) for i in db_rows.values], "agg_annotations_frame", 4)
    except sqlite3.Error as e:
        print(e)

    conn.commit()



if __name__ == "__main__":
    main()
