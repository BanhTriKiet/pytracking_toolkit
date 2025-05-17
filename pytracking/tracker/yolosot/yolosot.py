from pytracking.tracker.base import BaseTracker
from collections import OrderedDict
from ultralytics import YOLO
import numpy as np
import time
import cv2


class YOLOTracker(BaseTracker):
    def __init__(self, params):
        super().__init__(params)
        self.model = YOLO(self.params.model_path)
        self.initialized = False
        self.dataset_name = getattr(params, 'dataset_name', 'unknown')
        self.track_id_to_follow = None  # ID cần theo dõi

    def initialize(self, image, info: dict) -> dict:
        self.frame_num = 1
        if not self.params.has('device'):
            self.params.device = 'cuda' if self.params.use_gpu else 'cpu'

        # Time initialization
        tic = time.time()

        # Get initial bbox from info
        state = info['init_bbox']  # [x, y, w, h]
        self.prev_state = list(state)  # Keep in list for internal use

        # Initialize YOLO model
        self.model = YOLO(self.params.model_path)

        # Run YOLO tracking to get initial ID to follow
        result = self.model.track(image, persist=True, tracker="bytetrack.yaml", imgsz=320)[0]
        boxes = result.boxes

        min_id = float('inf')
        selected_box = None

        for box in boxes:
            if box.id is not None:
                tid = int(box.id.item()) if hasattr(box.id, "item") else int(box.id)
                if tid < min_id:
                    min_id = tid
                    selected_box = box

        if selected_box is not None:
            x1, y1, x2, y2 = selected_box.xyxy[0]
            self.prev_state = [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]
            self.track_id_to_follow = min_id
        else:
            print("[Warning] Không tìm thấy đối tượng ban đầu, dùng bbox khởi tạo.")

        # Save state as OrderedDict for return
        init_out = OrderedDict([
            ('x', self.prev_state[0]),
            ('y', self.prev_state[1]),
            ('w', self.prev_state[2]),
            ('h', self.prev_state[3])
        ])

        out = {
            'target_bbox': init_out,
            'time': time.time() - tic
        }
        return out


    def track(self, image, info: dict = None) -> dict:
        # Chạy YOLOv9 + ByteTrack để lấy kết quả
        result = self.model.track(image, persist=True, tracker="bytetrack.yaml", imgsz=320)[0]
        boxes = result.boxes
        matched_box = None
        min_id = float('inf')

        for box in boxes:
            if box.id is not None and len(box.id) > 0:
                tid = int(box.id)
                if tid < min_id:
                    min_id = tid
                    x1, y1, x2, y2 = box.xyxy[0]
                    matched_box = [
                        x1.item(),  # x
                        y1.item(),  # y
                        (x2 - x1).item(),  # width
                        (y2 - y1).item()   # height
                    ]

        if matched_box is not None:
            self.prev_state = matched_box
        else:
            print(f"[Warning] Không tìm thấy object nào trong frame {self.frame_num}, dùng prev_state.")

        return OrderedDict({'target_bbox': self.prev_state})
    @staticmethod
    def _compute_iou(boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]
        return interArea / float(boxAArea + boxBArea - interArea + 1e-6)
