from fastapi import FastAPI, File, UploadFile, Query
from pydantic import BaseModel
from typing import List
from sys import platform
import datetime, os, json, zlib

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *
import urllib
import numpy as np
import pandas as pd
import streamlit as st
import sqlite3, pims
import db_utils
from collections import OrderedDict
import skvideo.io


# Initialize API
app = FastAPI()
upload_folder = "/data/api"


class Item(BaseModel):
    file_data: str


@app.post("/create_file")
async def create_file(file: UploadFile = File(...)):
    global upload_folder
    file_object = file.file
    # create empty file to copy the file_object to
    upload_folder = open(os.path.join(upload_folder, file.filename), "wb+")
    shutil.copyfileobj(file_object, upload_folder)
    upload_folder.close()
    return {"filename": file.filename}


class KosterModel:
    def __init__(self):
        with torch.no_grad():
            self.img_size = (
                (320, 192) if ONNX_EXPORT else 416
            )  # (320, 192) or (416, 256) or (608, 352) for (height, width)
            self.out, self.weights, self.half, self.view_img, self.save_txt = (
                "/data/api",
                "/data/weights/last.pt",
                False,
                False,
                True,
            )
            self.webcam = False
            (
                self.names,
                self.conf_thres,
                self.classes,
                self.agnostic_nms,
                self.iou_thres,
            ) = ("data/koster.names", 0.3, None, False, 0.6)

            # Initialize
            self.device = torch_utils.select_device(
                device="cpu" if ONNX_EXPORT else "0"
            )
            if not os.path.exists(self.out):
                os.makedirs(self.out)  # make new output folder

            # Initialize model
            self.model = Darknet("cfg/yolov3-spp-1cls.cfg", self.img_size)

            # Load weights
            # attempt_download(self.weights)
            if self.weights.endswith(".pt"):  # pytorch format
                self.model.load_state_dict(
                    torch.load(self.weights, map_location=self.device)["model"]
                )
            else:  # darknet format
                load_darknet_weights(self.model, self.weights)

            # Eval mode
            self.model.to(self.device).eval()

            # Export mode
            if ONNX_EXPORT:
                model.fuse()
                img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
                f = opt.weights.replace(
                    opt.weights.split(".")[-1], "onnx"
                )  # *.onnx filename
                torch.onnx.export(model, img, f, verbose=False, opset_version=11)

                # Validate exported model
                import onnx

                model = onnx.load(f)  # Load the ONNX model
                onnx.checker.check_model(model)  # Check that the IR is well formed
                print(
                    onnx.helper.printable_graph(model.graph)
                )  # Print a human readable representation of the graph

            # Half precision
            self.half = (
                self.half and self.device.type != "cpu"
            )  # half precision only supported on CUDA
            if self.half:
                self.model.half()

    def detect(self, save_img=False):
        boxes = []
        vid = False
        detect_dict = {}
        my_bar = st.progress(0)
        vid_bar = st.progress(0)
        with torch.no_grad():
            # Set Dataloader
            vid_path, vid_writer = None, None
            if self.webcam:
                self.view_img = True
                torch.backends.cudnn.benchmark = (
                    True  # set True to speed up constant image size inference
                )
                dataset = LoadStreams(self.source, img_size=self.img_size)
            else:
                self.save_img = True
                dataset = LoadImages(self.source, img_size=self.img_size)

            # Get names and colors
            names = load_classes(self.names)
            colors = [
                [random.randint(0, 255) for _ in range(3)] for _ in range(len(names))
            ]
            # Initialize frame counter
            fcount = 0

            # Run inference
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:
                # Increment frame
                fcount += 1
                # Create output dict for each file
                if not path in detect_dict:
                    detect_dict[path] = []
                t = time.time()
                img = torch.from_numpy(img).to(self.device)
                img = img.half() if self.half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                pred = self.model(img)[0].float() if self.half else self.model(img)[0]

                # Apply NMS
                pred = non_max_suppression(
                    pred,
                    self.conf_thres,
                    self.iou_thres,
                    classes=self.classes,
                    agnostic=self.agnostic_nms,
                )

                # Apply Classifier
                classify = False
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Start progress counter
                i = 0

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if self.webcam:  # batch_size >= 1
                        p, s, im0 = path[i], "%g: " % i, im0s[i]
                    else:
                        p, s, im0 = path, "", im0s

                    save_path = str(Path(self.out) / Path(p).name)
                    s += "%gx%g " % img.shape[2:]  # print string
                    if det is not None and len(det):
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(
                            img.shape[2:], det[:, :4], im0.shape
                        ).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += "%g %ss, " % (n, names[int(c)])  # add to string

                        # Write results
                        for *xyxy, conf, cls in det:
                            boxes.append(xyxy)
                            if n.item() > 0:
                                detect_dict[path].append(
                                    [
                                        fcount,
                                        [i.item() for i in xyxy],
                                        cls.item(),
                                        conf.item(),
                                    ]
                                )
                            if self.save_txt:  # Write to file
                                with open(save_path + ".txt", "a") as file:
                                    file.write(("%g " * 6 + "\n") % (*xyxy, cls, conf))

                            if self.save_img or self.view_img:  # Add bbox to image
                                label = "%s %.2f" % (names[int(cls)], conf)
                                plot_one_box(
                                    xyxy, im0, label=label, color=colors[int(cls)]
                                )

                    # Print time (inference + NMS)
                    print("%sDone. (%.3fs)" % (s, time.time() - t))

                    # Stream results
                    if self.view_img:
                        cv2.imshow(p, im0)
                        if cv2.waitKey(1) == ord("q"):  # q to quit
                            raise StopIteration

                    # Save results (image with detections)
                    if self.save_img:
                        if dataset.mode == "images":
                            print("image")
                        #    cv2.imwrite(save_path, im0)
                        else:
                            vid = True
                            if vid_path != save_path:  # new video
                                vid_path = save_path
                                if isinstance(vid_writer, skvideo.io.FFmpegWriter):
                                    vid_writer.close()  # release previous video writer
                                if not os.path.isdir(Path(self.out) / "videos"):
                                    os.mkdir(Path(self.out) / "videos")
                                nvid_path = str(
                                    Path(self.out) / "videos" / Path(p).name
                                )
                                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                width = 416
                                r = width / float(w)
                                dim = (width, int(h * r))
                                vid_writer = skvideo.io.FFmpegWriter(
                                    nvid_path,
                                    inputdict={
                                        "-r": str(fps),
                                        "-s": "{}x{}".format(dim[0], dim[1]),
                                    },
                                    outputdict={
                                        "-r": str(fps),
                                        "-c:v": "libx264",
                                        "-crf": "17",
                                        "-preset": "ultrafast",
                                        "-pix_fmt": "yuv444p",
                                    },
                                )
                            vid_writer.writeFrame(
                                cv2.resize(
                                    cv2.cvtColor(im0, cv2.COLOR_RGB2BGR),
                                    dim,
                                    interpolation=cv2.INTER_AREA,
                                )
                            )

            i += 1
            perc_complete = i / len(dataset)
            my_bar.progress(perc_complete)
            if vid:
                vid_writer.close()

            if self.save_txt or self.save_img:
                print("Results saved to %s" % os.getcwd() + os.sep + self.out)
                if platform == "darwin":  # MacOS
                    os.system("open " + self.out + " " + save_path)

            print("Done. (%.3fs)" % (time.time() - t0))
            if vid:
                # Return the path only if frontend and backend don't require transfers
                return nvid_path, vid, detect_dict
            # open(nvid_path, "rb").read(), vid, detect_dict
            else:
                # Compress image before transfer
                width = 416
                (h, w) = im0.shape[:2]
                r = width / float(w)
                dim = (width, int(h * r))
                # resize the image
                resized = cv2.resize(im0, dim, interpolation=cv2.INTER_AREA)
                return resized, vid, detect_dict


# Initialize model
model = KosterModel()

# Sanity request check
@app.get("/ping")
def ping():
    return {"message": "pong! your requested was heard"}


# Main prediction function
@app.post("/predict")
async def predict(media_path: str, conf_thres: float, iou_thres: float):
    try:
        media_path = await create_file(media_path)
    except:
        pass
    model.source = media_path
    model.conf_thres = conf_thres
    model.iou_thres = iou_thres
    pred, vid, detect_dict = model.detect()
    if vid:
        pred = list(pred)
    else:
        pred = pred.tolist()
    return {"vid": vid, "prediction": pred, "prediction_dict": detect_dict}


# Loading uploaded videos from database utility function
@app.get("/data")
async def load_data():
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

    return {"data": df.to_dict()}


# Utility function for single frame
@app.get("/read")
async def get_movie_frame(file_path: str, frame_number: int):
    return {
        "frame_data": json.dumps(np.array(pims.Video(file_path)[frame_number]).tolist())
    }


@app.post("/save")
async def save_image(file_name: str, file_data=File(...)):
    if not os.path.isfile(f"{model.out}/{file_name}"):
        with open(f"{model.out}/{file_name}", "wb") as out_file:
            out_file.write(file_data.file.read())
    return {"output": f"{model.out}/{file_name}"}


@app.post("/save_vid")
async def save_video(file_name: str, fps: int, w: int, h: int, file_data=File(...)):
    if not os.path.isfile(f"{model.out}/{file_name}"):
        with open(f"{model.out}/{file_name}", "wb") as out_file:
            decompressed = zlib.decompress(file_data.file.read())
            out_file.write(decompressed)
    return {"output": f"{model.out}/{file_name}"}
