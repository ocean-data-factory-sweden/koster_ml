from fastapi import FastAPI
from sys import platform

from models import *  # set ONNX_EXPORT in models.py
from utils.datasets import *
from utils.utils import *

app = FastAPI()

Class KosterModel():

	def __init__(self, config):
		with torch.no_grad():
			self.img_size = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
		    self.out, self.source, self.weights, self.half, self.view_img, self.save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
		    self.webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt')
		    self.names, self.conf_thres, self.classes, self.agnostic_nms = opt.names, opt.conf_thres, opt.classes, opt.agnostic_nms

		    # Initialize
		    self.device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
		    if os.path.exists(out):
		        shutil.rmtree(out)  # delete output folder
		    os.makedirs(out)  # make new output folder

		    # Initialize model
		    self.model = Darknet(opt.cfg, self.img_size)

		    # Load weights
		    attempt_download(self.weights)
		    if self.weights.endswith('.pt'):  # pytorch format
		        self.model.load_state_dict(torch.load(self.weights, map_location=self.device)['model'])
		    else:  # darknet format
		        load_darknet_weights(self.model, self.weights)

		    # Second-stage classifier
		    classify = False
		    if classify:
		        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
		        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
		        modelc.to(device).eval()

		    # Fuse Conv2d + BatchNorm2d layers
		    # model.fuse()
		    # torch_utils.model_info(model, report='summary')  # 'full' or 'summary'

		    # Eval mode
		    self.model.to(self.device).eval()

		    # Export mode
		    if ONNX_EXPORT:
		        model.fuse()
		        img = torch.zeros((1, 3) + img_size)  # (1, 3, 320, 192)
		        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
		        torch.onnx.export(model, img, f, verbose=False, opset_version=11)

		        # Validate exported model
		        import onnx
		        model = onnx.load(f)  # Load the ONNX model
		        onnx.checker.check_model(model)  # Check that the IR is well formed
		        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
		        return

		    # Half precision
		    self.half = self.half and self.device.type != 'cpu'  # half precision only supported on CUDA
		    if self.half:
		        self.model.half()


	def detect(self, save_img=False):

	    with torch.no_grad():
		    # Set Dataloader
		    vid_path, vid_writer = None, None
		    if self.webcam:
		        self.view_img = True
		        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
		        dataset = LoadStreams(self.source, img_size=self.img_size)
		    else:
		        save_img = True
		        dataset = LoadImages(self.source, img_size=self.img_size)

		    # Get names and colors
		    names = load_classes(self.names)
		    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

		    # Run inference
		    t0 = time.time()
		    for path, img, im0s, vid_cap in dataset:
		        t = time.time()
		        img = torch.from_numpy(img).to(self.device)
		        img = img.half() if half else img.float()  # uint8 to fp16/32
		        img /= 255.0  # 0 - 255 to 0.0 - 1.0
		        if img.ndimension() == 3:
		            img = img.unsqueeze(0)

		        # Inference
		        pred = self.model(img)[0].float() if half else self.model(img)[0]

		        # Apply NMS
		        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

		        # Apply Classifier
		        if classify:
		            pred = apply_classifier(pred, modelc, img, im0s)

		        # Process detections
		        for i, det in enumerate(pred):  # detections per image
		            if webcam:  # batch_size >= 1
		                p, s, im0 = path[i], '%g: ' % i, im0s[i]
		            else:
		                p, s, im0 = path, '', im0s

		            save_path = str(Path(self.out) / Path(p).name)
		            s += '%gx%g ' % img.shape[2:]  # print string
		            if det is not None and len(det):
		                # Rescale boxes from img_size to im0 size
		                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

		                # Print results
		                for c in det[:, -1].unique():
		                    n = (det[:, -1] == c).sum()  # detections per class
		                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

		                # Write results
		                for *xyxy, conf, cls in det:
		                    if save_txt:  # Write to file
		                        with open(save_path + '.txt', 'a') as file:
		                            file.write(('%g ' * 6 + '\n') % (*xyxy, cls, conf))

		                    if save_img or view_img:  # Add bbox to image
		                        label = '%s %.2f' % (names[int(cls)], conf)
		                        plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])

		            # Print time (inference + NMS)
		            print('%sDone. (%.3fs)' % (s, time.time() - t))

		            # Stream results
		            if view_img:
		                cv2.imshow(p, im0)
		                if cv2.waitKey(1) == ord('q'):  # q to quit
		                    raise StopIteration

		            # Save results (image with detections)
		            if save_img:
		                if dataset.mode == 'images':
		                    cv2.imwrite(save_path, im0)
		                else:
		                    if vid_path != save_path:  # new video
		                        vid_path = save_path
		                        if isinstance(vid_writer, cv2.VideoWriter):
		                            vid_writer.release()  # release previous video writer

		                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
		                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
		                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
		                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
		                    vid_writer.write(im0)

		    if save_txt or save_img:
		        print('Results saved to %s' % os.getcwd() + os.sep + out)
		        if platform == 'darwin':  # MacOS
		            os.system('open ' + out + ' ' + save_path)

		    print('Done. (%.3fs)' % (time.time() - t0))

@app.get("/ping")
def ping():
    return {"message": "pong!"}


@app.get("/predict/{media_path}")
def predict(media_path: str):
	m = KosterModel()
    pred = m.detect(media_path)
    return {"message": str(pred)}




