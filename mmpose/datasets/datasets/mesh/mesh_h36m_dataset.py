import copy as cp
import os
from collections import OrderedDict

import json_tricks as json
import numpy as np

from mmpose.datasets.builder import DATASETS
from .mesh_base_dataset import MeshBaseDataset


def compute_similarity_transform(S1, S2):
    """Computes a similarity transform (sR, t) that takes a set of 3D points S1
    (N x 3) closest to a set of 3D points S2, where R is an 3x3 rotation
    matrix, t 3x1 translation, s scale. And return the transformed 3D points
    S1_hat (N x 3). i.e. solves the orthogonal Procrutes problem.

    Notes:
        Points number: N

    Args:
        S1 (np.ndarray([N, 3])): Source point set.
        S2 (np.ndarray([N, 3])): Target point set.

    Returns:
        S1_hat (np.ndarray([N, 3])): Transformed source point set.
    """

    transposed = False
    if S1.shape[0] != 3 and S1.shape[0] != 2:
        S1 = S1.T
        S2 = S2.T
        transposed = True
    assert (S2.shape[1] == S1.shape[1])

    # 1. Remove mean.
    mu1 = S1.mean(axis=1, keepdims=True)
    mu2 = S2.mean(axis=1, keepdims=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    # 2. Compute variance of X1 used for scale.
    var1 = np.sum(X1**2)

    # 3. The outer product of X1 and X2.
    K = X1.dot(X2.T)

    # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
    # singular vectors of K.
    U, s, Vh = np.linalg.svd(K)
    V = Vh.T
    # Construct Z that fixes the orientation of R to get det(R)=1.
    Z = np.eye(U.shape[0])
    Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
    # Construct R.
    R = V.dot(Z.dot(U.T))

    # 5. Recover scale.
    scale = np.trace(R.dot(K)) / var1

    # 6. Recover translation.
    t = mu2 - scale * (R.dot(mu1))

    # 7. Transform the source points:
    S1_hat = scale * R.dot(S1) + t

    if transposed:
        S1_hat = S1_hat.T

    return S1_hat


@DATASETS.register_module()
class MeshH36MDataset(MeshBaseDataset):
    """Human3.6M Dataset dataset for 3D human mesh estimation.

    The dataset loads raw features and apply specified transforms
    to return a dict containing the image tensors and other information.

    Args:
        ann_file (str): Path to the annotation file.
        img_prefix (str): Path to a directory where images are held.
            Default: None.
        data_cfg (dict): config
        pipeline (list[dict | callable]): A sequence of data transforms.
        test_mode (bool): Store True when building test or
            validation dataset. Default: False.
    """

    def __init__(self,
                 ann_file,
                 img_prefix,
                 data_cfg,
                 pipeline,
                 test_mode=False):

        super().__init__(
            ann_file, img_prefix, data_cfg, pipeline, test_mode=test_mode)

        # flip_pairs in Human3.6M.
        # For all mesh dataset, we use 24 joints as CMR and SPIN.
        self.ann_info['flip_pairs'] = [[0, 5], [1, 4], [2, 3], [6, 11],
                                       [7, 10], [8, 9], [20, 21], [22, 23]]

        # origin_part:  [0, 1, 2, 3, 4, 5, 6,  7,  8, 9, 10,11, 12, 13,
        # 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]
        # flipped_part: [5, 4, 3, 2, 1, 0, 11, 10, 9, 8, 7, 6,  12, 13,
        # 14, 15, 16, 17, 18, 19, 21, 20, 23, 22]

        self.ann_info['use_different_joints_weight'] = False
        self.ann_info['joints_weight'] =  \
            np.ones(24, dtype=np.float32).reshape(
                (self.ann_info['num_joints'], 1))

        self.ann_info['uv_type'] = data_cfg['uv_type']
        self.ann_info['use_IUV'] = data_cfg['use_IUV']
        self.iuv_prefix = os.path.join(
            self.img_prefix, '{}_IUV_gt'.format(self.ann_info['uv_type']))
        self.db = self._get_db(ann_file)

    def _get_db(self, ann_file):
        """Load dataset."""
        data = np.load(ann_file)
        tmpl = dict(
            image_file=None,
            center=None,
            scale=None,
            rotation=0,
            joints_2d=None,
            joints_2d_visible=None,
            joints_3d=None,
            joints_3d_visible=None,
            gender=None,
            pose=None,
            beta=None,
            has_smpl=0,
            iuv_file=None,
            has_iuv=0,
            dataset='H36M')
        gt_db = []

        imgnames_ = data['imgname']
        scales_ = data['scale']
        centers_ = data['center']
        dataset_len = len(imgnames_)

        # Get 2D keypoints
        try:
            keypoints_ = data['part']
        except KeyError:
            keypoints_ = np.zeros((dataset_len, 24, 3), dtype=np.float)

        # Get gt 3D joints, if available
        try:
            joints_3d_ = data['S']
        except KeyError:
            joints_3d_ = np.zeros((dataset_len, 24, 4), dtype=np.float)

        # Get gt SMPL parameters, if available
        try:
            poses_ = data['pose'].astype(np.float)
            betas_ = data['shape'].astype(np.float)
            has_smpl = 1
        except KeyError:
            poses_ = np.zeros((dataset_len, 72), dtype=np.float)
            betas_ = np.zeros((dataset_len, 10), dtype=np.float)
            has_smpl = 0

        # Get gender data, if available
        try:
            genders_ = data['gender']
            genders_ = np.array([0 if str(g) == 'm' else 1
                                 for g in genders_]).astype(np.int32)
        except KeyError:
            genders_ = -1 * np.ones(dataset_len).astype(np.int32)

        # Get IUV image, if available
        try:
            iuv_names_ = data['iuv_names']
            has_iuv = has_smpl
        except KeyError:
            iuv_names_ = [''] * dataset_len
            has_iuv = 0

        for i in range(len(data['imgname'])):
            newitem = cp.deepcopy(tmpl)
            newitem['image_file'] = os.path.join(self.img_prefix, imgnames_[i])
            # newitem['scale'] = scales_[i].item()
            newitem['scale'] = self.ann_info['image_size'] / scales_[i].item(
            ) / 200.0
            newitem['center'] = centers_[i]

            newitem['joints_2d'] = keypoints_[i, :, :2]
            newitem['joints_2d_visible'] = keypoints_[i, :, -1][:, np.newaxis]
            newitem['joints_3d'] = joints_3d_[i, :, :3]
            newitem['joints_3d_visible'] = keypoints_[i, :, -1][:, np.newaxis]
            newitem['pose'] = poses_[i]
            newitem['beta'] = betas_[i]
            newitem['has_smpl'] = has_smpl
            newitem['gender'] = genders_[i]
            newitem['iuv_file'] = os.path.join(self.iuv_prefix, iuv_names_[i])
            newitem['has_iuv'] = has_iuv
            gt_db.append(newitem)
        return gt_db

    def evaluate(self, outputs, res_folder, metric='joint_error', **kwargs):
        """Evaluate 3D keypoint results."""
        assert metric == 'joint_error'

        res_file = os.path.join(res_folder, 'result_keypoints.json')
        all_preds, all_boxes, all_image_path = list(map(list, zip(*outputs)))

        kpts = []
        for idx, kpt in enumerate(all_preds):
            kpts.append({
                'joints_3d': kpt[0],
                'smpl_pose': kpt[1],
                'smpl_beta': kpt[2],
                'center': all_boxes[idx][0][0:2],
                'scale': all_boxes[idx][0][2:4],
                'area': all_boxes[idx][0][4],
                'score': all_boxes[idx][0][5],
                'image': int(all_image_path[idx][-13:-4]),
            })

        self._write_keypoint_results(kpts, res_file)
        info_str = self._report_metric(res_file)
        name_value = OrderedDict(info_str)
        return name_value

    def _write_keypoint_results(self, keypoints, res_file):
        """Write results into a json file."""

        with open(res_file, 'w') as f:
            json.dump(keypoints, f, sort_keys=True, indent=4)

    def _report_metric(self, res_file):
        """Keypoint evaluation.

        Report mean per joint position error (MPJPE) and mean per joint
        position error after rigid alignment (MPJPE-PA)
        """

        with open(res_file, 'r') as fin:
            preds = json.load(fin)
        assert len(preds) == len(self.db)

        joint_error = []
        joint_error_pa = []

        for pred, item in zip(preds, self.db):
            error, error_pa = self.evaluate_kernal(pred['joints3d'],
                                                   item['joints_3d'],
                                                   item['joints_3d_visible'])
            joint_error.append(error)
            joint_error_pa.append(error_pa)

        mpjpe = joint_error.mean()
        mpjpe_pa = joint_error_pa.mean()

        info_str = []
        info_str.append(('MPJPE', mpjpe))
        info_str.append(('MPJPE-PA', mpjpe_pa))
        return info_str

    def evaluate_kernal(self, pred, joints_3d, joints_3d_visible):
        """Evaluate one example."""
        # Only 14 lsp joints are used for evaluation
        joint_mapper = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18]

        assert (joints_3d_visible[joint_mapper].min() > 0)

        pred_joints_3d = pred[joint_mapper, :]
        pred_pelvis = (pred_joints_3d[:, [2]] + pred_joints_3d[:, [3]]) / 2
        pred_joints_3d = pred_joints_3d - pred_pelvis

        gt_joints_3d = joints_3d[joint_mapper, :]
        gt_pelvis = (gt_joints_3d[:, [2]] + gt_joints_3d[:, [3]]) / 2
        gt_joints_3d = gt_joints_3d - gt_pelvis

        error = pred_joints_3d - gt_joints_3d
        error = np.sqrt((error**2).sum(axis=-1)).mean(axis=-1)

        pred_joints_3d_aligned = compute_similarity_transform(
            pred_joints_3d, gt_joints_3d)
        error_pa = pred_joints_3d_aligned - gt_joints_3d
        error_pa = np.sqrt((error_pa**2).sum(axis=-1)).mean(axis=-1)

        return error, error_pa
