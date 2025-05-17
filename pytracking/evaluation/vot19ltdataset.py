import numpy as np
from pytracking.evaluation.data import Sequence, BaseDataset, SequenceList
import os
import glob


class VOTLT2019Dataset(BaseDataset):
    """VOT-LT2019 dataset."""
    def __init__(self):
        super().__init__()
        self.base_path = self.env_settings.votlt2019_path  # phải thêm vào local.py
        self.sequence_list = self._get_sequence_list()

    def get_sequence_list(self):
        return SequenceList([self._construct_sequence(s) for s in self.sequence_list])

    def _construct_sequence(self, sequence_name):
        seq_path = os.path.join(self.base_path, sequence_name)
        gt_path = os.path.join(seq_path, 'groundtruth.txt')
        ground_truth_rect = np.loadtxt(gt_path, delimiter=',', dtype=np.float64)

        # Sửa tại đây: đọc ảnh từ thư mục 'color'
        color_dir = os.path.join(seq_path, 'color')
        img_files = sorted(glob.glob(os.path.join(color_dir, '*.jpg')))

        return Sequence(sequence_name, img_files, 'votlt2019', ground_truth_rect)

    def __len__(self):
        return len(self.sequence_list)

    def _get_sequence_list(self):
        # lấy tất cả thư mục có chứa groundtruth.txt
        return sorted([f for f in os.listdir(self.base_path)
                       if os.path.isdir(os.path.join(self.base_path, f))
                       and os.path.isfile(os.path.join(self.base_path, f, 'groundtruth.txt'))])
