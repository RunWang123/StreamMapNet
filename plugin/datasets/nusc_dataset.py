from.base_dataset import BaseMapDataset
from .map_utils.nuscmap_extractor import NuscMapExtractor
from mmdet.datasets import DATASETS
import numpy as np
from .visualize.renderer import Renderer
import mmcv
from time import time
from pyquaternion import Quaternion
import math

@DATASETS.register_module()
class NuscDataset(BaseMapDataset):
    """NuScenes map dataset class.

    Args:
        ann_file (str): annotation file path
        cat2id (dict): category to class id
        roi_size (tuple): bev range
        eval_config (Config): evaluation config
        meta (dict): meta information
        pipeline (Config): data processing pipeline config
        interval (int): annotation load interval
        work_dir (str): path to work dir
        test_mode (bool): whether in test mode
    """
    
    def __init__(self, data_root, **kwargs):
        super().__init__(**kwargs)
        self.map_extractor = NuscMapExtractor(data_root, self.roi_size)
        self.renderer = Renderer(self.cat2id, self.roi_size, 'nusc')
    
    def load_annotations(self, ann_file):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations.
        """
        
        
        start_time = time()
        ann = mmcv.load(ann_file)
        
        # Handle both formats:
        # - Official format: direct list
        # - Temporal format: dict with 'infos' key
        if isinstance(ann, dict) and 'infos' in ann:
            samples = ann['infos'][::self.interval]
        else:
            samples = ann[::self.interval]
        
        print(f'collected {len(samples)} samples in {(time() - start_time):.2f}s')
        self.samples = samples

    def get_sample(self, idx):
        """Get data sample. For each sample, map extractor will be applied to extract 
        map elements. 

        Args:
            idx (int): data index

        Returns:
            result (dict): dict of input
        """

        sample = self.samples[idx]
        
        # Handle key name differences between formats
        # Official: 'location', Temporal: 'map_location'
        location = sample.get('location', sample.get('map_location'))
        # Official: 'e2g_translation', Temporal: 'ego2global_translation'
        e2g_translation = sample.get('e2g_translation', sample.get('ego2global_translation'))
        # Official: 'e2g_rotation', Temporal: 'ego2global_rotation'
        e2g_rotation = sample.get('e2g_rotation', sample.get('ego2global_rotation'))
        
        map_geoms = self.map_extractor.get_map_geom(location, e2g_translation, e2g_rotation)

        map_label2geom = {}
        for k, v in map_geoms.items():
            if k in self.cat2id.keys():
                map_label2geom[self.cat2id[k]] = v
        
        ego2img_rts = []
        cam_intrinsics_list = []
        cam_extrinsics_list = []
        img_filenames_list = []
        
        for c in sample['cams'].values():
            # Handle intrinsics: Official: 'intrinsics', Temporal: 'cam_intrinsic'
            intrinsic = np.array(c.get('intrinsics', c.get('cam_intrinsic')))
            cam_intrinsics_list.append(intrinsic)
            
            # Handle extrinsics
            if 'extrinsics' in c:
                extrinsic = np.array(c['extrinsics'])
            else:
                # Temporal format: compute ego2cam from sensor2ego
                # sensor2ego gives us cam2ego transformation
                # We need ego2cam, which is the inverse
                from scipy.spatial.transform import Rotation as R
                sensor2ego_rot = c['sensor2ego_rotation']
                sensor2ego_trans = c['sensor2ego_translation']
                
                # Convert quaternion [w, x, y, z] to [x, y, z, w] for scipy
                quat_xyzw = [sensor2ego_rot[1], sensor2ego_rot[2], sensor2ego_rot[3], sensor2ego_rot[0]]
                cam2ego_rot = R.from_quat(quat_xyzw).as_matrix()
                cam2ego_trans = np.array(sensor2ego_trans)
                
                # Invert to get ego2cam
                ego2cam_rot = cam2ego_rot.T
                ego2cam_trans = -ego2cam_rot @ cam2ego_trans
                
                extrinsic = np.eye(4)
                extrinsic[:3, :3] = ego2cam_rot
                extrinsic[:3, 3] = ego2cam_trans
            
            cam_extrinsics_list.append(extrinsic)
            
            ego2cam_rt = extrinsic
            viewpad = np.eye(4)
            viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
            ego2cam_rt = (viewpad @ ego2cam_rt)
            ego2img_rts.append(ego2cam_rt)
            
            # Handle image path: Official: 'img_fpath', Temporal: 'data_path'
            img_path = c.get('img_fpath', c.get('data_path'))
            # Temporal format may have relative paths like './data/nuscenes/samples/...'
            if img_path and not img_path.startswith('/'):
                if 'samples/' in img_path:
                    # Extract 'samples/...' part for temporal format
                    import os
                    data_root = '/home/runw/Project/data/mini/nuscenes/'
                    img_path = os.path.join(data_root, img_path[img_path.index('samples/'):])
            img_filenames_list.append(img_path)

        # if sample['sample_idx'] == 0:
        #     is_first_frame = True
        # else:
        #     is_first_frame = self.flag[sample['sample_idx']] > self.flag[sample['sample_idx'] - 1]
        input_dict = {
            'location': location,
            'token': sample['token'],
            'img_filenames': img_filenames_list,
            # intrinsics are 3x3 Ks
            'cam_intrinsics': cam_intrinsics_list,
            # extrinsics are 4x4 tranform matrix, **ego2cam**
            'cam_extrinsics': cam_extrinsics_list,
            'ego2img': ego2img_rts,
            'map_geoms': map_label2geom, # {0: List[ped_crossing(LineString)], 1: ...}
            'ego2global_translation': e2g_translation, 
            'ego2global_rotation': Quaternion(e2g_rotation).rotation_matrix.tolist(),
            # 'is_first_frame': is_first_frame, # deprecated
            # Official: 'sample_idx', Temporal: 'frame_idx'
            'sample_idx': sample.get('sample_idx', sample.get('frame_idx', 0)),
            # Official: 'scene_name', Temporal: 'scene_token'
            'scene_name': sample.get('scene_name', sample.get('scene_token', ''))
            # 'group_idx': self.flag[sample['sample_idx']]
        }

        return input_dict