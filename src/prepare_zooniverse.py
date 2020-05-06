# module imports
import os, json, argparse
import pandas as pd
import numpy as np
import frame_tracker
from collections import OrderedDict
from pathlib import Path
from ast import literal_eval
from scipy.spatial.distance import pdist

from PIL import Image
from mydia import Videos
from db_utils import create_connection

# utility functions


def get_id(conn, row):

    # Currently we discard sites that have no lat or lon coordinates, since site descriptions are not unique
    # it becomes difficult to match this information otherwise
    try:
        gid = pd.read_sql_query(
            f"SELECT id FROM species WHERE label=='{row['class_name']}'", conn
        ).values[0][0]
        print(gid)
    except IndexError:
        gid = 0
    return gid


def extract_frames(filenames, frame_numbers, out_path):
    # read all videos
    reader = Videos()
    videos = reader.read(list(set(filenames)), workers=8)
    m_names = [
        os.path.basename(os.path.splitext(x)[0]) if isinstance(x, str) else x
        for x in list(set(filenames))
    ]

    for i in range(len(videos)):
        Image.fromarray(videos[i, frame_numbers[i], ...]).save(
            f"{out_path}/{m_names[i]}_frame_{frame_numbers[i]}.jpeg"
        )

    print("Frames extracted successfully")
    return None


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


def bb_iou(boxA, boxB):

    # Compute edges
    boxA[2], boxA[3] = boxA[0] + boxA[2], boxA[1] + boxA[3]
    boxB[2], boxB[3] = boxB[0] + boxB[2], boxB[1] + boxB[3]

    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def filter_bboxes(bboxes, agg="mean", threshold=0.2):
    # If there is only one annotation, we take it as the truth (at least for testing)
    if len(bboxes) <= 1:
        return bboxes
    dm = pdist(bboxes, bb_iou)
    dm.resize((len(dm), len(dm)), refcheck=False)
    dm = np.fliplr(dm)
    dm = dm + dm.T - np.diag(np.diag(dm))
    valid = np.argwhere(dm.mean(axis=1) >= threshold)
    if agg == "mean":
        return bboxes[valid].mean(axis=0)


def main():
    "Handles argument parsing and launches the correct function."
    parser = argparse.ArgumentParser()
    # parser.add_argument(
    #    "--w2_path",
    #    "-w2",
    #    help="output from workflow 2 in Zooniverse",
    #    type=str,
    #    required=True,
    # )
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

    # train_rows = pd.read_sql_query(f"SELECT * FROM agg_annotations_frame WHERE species_id IN {species_ref}", conn)
    # test using cv
    train_rows = prepare(
        "../../../database/koster_lab_development/data_example/workflow2_classifications.csv"
    )

    bboxes = {}
    tboxes = {}
    new_rows = []

    for name, group in train_rows.groupby(["filename", "class_name", "start_frame"]):

        filename, class_name, start_frame = name

        # temp for testing
        filename = "../data/data_example/test_video.mp4"

        # Filter bboxes using IOU metric (essentially a consensus metric)
        # Keep only bboxes where mean overlap exceeds this threshold
        group = filter_bboxes(
            bboxes=[np.array((i[3], i[4], i[5], i[6])) for i in group.values]
        )

        # Track intermediate frames
        bboxes[name], tboxes[name] = [], []
        bboxes[name].extend(tuple(i) for i in group)
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

        if not os.path.isdir(args.out_path):
            os.mkdir(args.out_path)

        if args.out_format == "yolo":
            open(f"{args.out_path}/{file_base}_frame_{name[1]}.txt", "w").write(
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

    extract_frames(
        txt_rows.obj["filename"].tolist(), txt_rows.obj["frame"].tolist(), args.out_path
    )


if __name__ == "__main__":
    main()
