import argparse
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

import pandas as pd
import datetime
import db_utils


def detect(save_img=False):
    img_size = (
        (320, 192) if ONNX_EXPORT else opt.img_size
    )  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt, save_obs, db_path = (
        opt.output,
        opt.source,
        opt.weights,
        opt.half,
        opt.view_img,
        opt.save_txt,
        opt.save_obs,
        opt.db_path,
    )
    webcam = (
        source == "0"
        or source.startswith("rtsp")
        or source.startswith("http")
        or source.endswith(".txt")
    )

    # Initialize
    device = torch_utils.select_device(device="cpu" if ONNX_EXPORT else opt.device)
    if not os.path.exists(out):
        os.makedirs(out)  # make new output folder

    model_tuple = (
        opt.cfg,
        opt.conf_thres,
        opt.img_size,
        opt.iou_thres,
        opt.names,
        opt.weights,
    )

    # Check if model exists
    model_id = db_utils.get_id(
        "id",
        "models",
        db_utils.create_connection(db_path),
        conditions={
            "config_file": f"={opt.cfg}",
            "conf_thres": f"={opt.conf_thres}",
            "img_size": f"={opt.img_size}",
            "iou_thres": f"={opt.iou_thres}",
            "names_file": f"={opt.names}",
            "weights_file": f"={opt.weights}",
        },
    )
    # If not already in models, add to models and get model_id
    if not model_id:
        db_utils.add_to_table(
            db_path, "models", [(None,) + model_tuple], 7,
        )
        model_id = pd.read_sql_query(
            "SELECT MAX(id) FROM models", db_utils.create_connection(db_path)
        ).iloc[0]

    # Timestamp detections
    now = datetime.datetime.today()
    nTime = now.strftime("%d-%m-%Y-%H-%M-%S")
    dest = os.path.join(out + "/" + nTime)
    if not os.path.exists(dest):
        os.makedirs(dest)  # create dest dir
    # Initialize model
    model = Darknet(opt.cfg, img_size)

    # Load weights
    attempt_download(weights)
    if weights.endswith(".pt"):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)["model"])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name="resnet101", n=2)  # initialize
        modelc.load_state_dict(
            torch.load("weights/resnet101.pt", map_location=device)["model"]
        )  # load weights
        modelc.to(device).eval()

    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()
    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

    # Eval mode
    model.to(device).eval()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split(".")[-1], "onnx")  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

        # Validate exported model
        import onnx

        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(
            onnx.helper.printable_graph(model.graph)
        )  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != "cpu"  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = True
        torch.backends.cudnn.benchmark = (
            True  # set True to speed up constant image size inference
        )
        dataset = LoadStreams(source, img_size=img_size)
    else:
        save_img = True
        dataset = LoadImages(source, img_size=img_size)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    paths, n_observations = [], []
    t0 = time.time()
    fid = 0
    for path, img, im0s, vid_cap in dataset:
        paths.append(path)
        # get movie_id when using previously uploaded footage
        try:
            movie_id = db_utils.get_id(
                "id",
                "movies",
                db_utils.create_connection(db_path),
                conditions={"fpath": f"={path}"},
            )
        except:
            movie_id = 9999999
        t = time.time()
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        pred = model(img)[0].float() if half else model(img)[0]

        # Apply NMS
        pred = non_max_suppression(
            pred,
            opt.conf_thres,
            opt.iou_thres,
            classes=opt.classes,
            agnostic=opt.agnostic_nms,
        )

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], "%g: " % i, im0s[i]
            else:
                p, s, im0 = path, "", im0s

            save_path = str(Path(dest) / Path(p).name)
            s += "%gx%g " % img.shape[2:]  # print string
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    n_observations.append(n.item())
                    s += "%g %ss, " % (n, names[int(c)])  # add to string

                # Write results
                for *xyxy, conf, cls in det:
                    if save_txt:  # Write to file
                        with open(save_path + ".txt", "a") as file:
                            file.write(("%g " * 6 + "\n") % (*xyxy, cls, conf))
                    if save_obs:  # Add to database
                        # TODO Jannes: remove hard-coded species_id
                        db_utils.add_to_table(
                            db_path,
                            "model_annotations",
                            [(None,) + (model_id, i, 8, movie_id, now, *xyxy, conf)],
                            7,
                        )

                    if save_img or view_img:  # Add bbox to image
                        label = "%s %.2f" % (names[int(cls)], conf)
                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

            elif det is None or len(det) == 0:
                n_observations.append(0)

        # Export 10,000 frames per file to csv
        if save_obs:
            save_img = False
            # Save model configuration
            with open(dest + "/model_config.txt", "w") as file:
                file.write(str(opt))

            if len(paths) == 10000:
                obs_df = pd.DataFrame(
                    np.column_stack([paths, n_observations]), columns=["path", "n"]
                )
                obs_df.to_csv(dest + f"/{fid}_obs_summary.csv")
                fid += 1
                paths, n_observations = [], []

            # Print time (inference + NMS)
            print("%sDone. (%.3fs)" % (s, time.time() - t))

            # Stream results
            if view_img:
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord("q"):  # q to quit
                    raise StopIteration

            # Save results (image with detections)
            if save_img:
                if dataset.mode == "images":
                    cv2.imwrite(save_path, im0)
                else:
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer

                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        vid_writer = cv2.VideoWriter(
                            save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h)
                        )
                    vid_writer.write(im0)

    if save_txt or save_img:
        print("Results saved to %s" % os.getcwd() + os.sep + out)
        if platform == "darwin":  # MacOS
            os.system("open " + out + " " + save_path)

    print("Done. (%.3fs)" % (time.time() - t0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cfg", type=str, default="cfg/yolov3-spp.cfg", help="*.cfg path"
    )
    parser.add_argument(
        "--names", type=str, default="data/coco.names", help="*.names path"
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolov3-spp-ultralytics.pt",
        help="weights path",
    )
    parser.add_argument(
        "--source", type=str, default="data/samples", help="source"
    )  # input file/folder, 0 for webcam
    parser.add_argument(
        "--output", type=str, default="output", help="output folder"
    )  # output folder
    parser.add_argument(
        "--img-size", type=int, default=416, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.3, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.6, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--fourcc",
        type=str,
        default="mp4v",
        help="output video codec (verify ffmpeg support)",
    )   
    parser.add_argument(
        "--half", action="store_true", help="half precision FP16 inference"
    )
    parser.add_argument("--device", default="", help="device id (i.e. 0 or 0,1) or cpu")
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument("--classes", nargs="+", type=int, help="filter by class")
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument(
        "--save-obs", action="store_true", help="saving observations for visualisation"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/data/database/demo.db",
        help="database path to store new model observations",
    )
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        detect()
