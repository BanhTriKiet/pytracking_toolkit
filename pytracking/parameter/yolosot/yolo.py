from pytracking.utils import TrackerParams

def parameters():
    params = TrackerParams()

    # General tracking parameters
    params.debug = 0
    params.visualization = True
    params.use_gpu = False
    params.dataset_name = 'vot19lt'
    # Example YOLO-related config (cần tương thích với tracker/yolosot/yolotracker.py)
    params.model_path = 'D:\\CacMonHoc\\Nam4\\KLTN\\pytracking\\pytracking\\tracker\\yolosot\\v9_MOT17_320_200.pt'  # Đường dẫn đến YOLOv9 model
    # params.conf_threshold = 0.3
    # params.iou_threshold = 0.5

    # # Các tham số giả định khác
    # params.search_area_scale = 4.0
    # params.image_sample_size = 512

    return params
