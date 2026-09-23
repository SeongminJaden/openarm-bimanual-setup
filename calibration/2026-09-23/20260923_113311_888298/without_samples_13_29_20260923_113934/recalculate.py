"""Recompute without original samples 13/29; preserve original holdout membership."""
import json, sys
from pathlib import Path
import cv2
import numpy as np
from scipy.spatial.transform import Rotation
sys.path.insert(0, '/home/seongmin/ros2_ws/src/openarm_vision_pick')
from openarm_vision_pick.chest_calibration import checked_pose, pose, difference, write_json
source = Path('/home/seongmin/chest_calibration/20260923_113311_888298')
folder = Path(__file__).resolve().parent
data = json.loads((source / 'samples.json').read_text())
previous = json.loads((source / 'result.json').read_text())
samples = data['samples']
assert len(samples) == previous['sample_count'] == 31, 'Source session changed'
hands = [checked_pose(s['base_hand']) for s in samples]
markers = [checked_pose(s['camera_marker']) for s in samples]
validation = [i for i in range(31) if i % 5 == 4]
original_train = [i for i in range(31) if i not in validation]
excluded = {12, 28}  # one-based samples 13, 29
train = [i for i in original_train if i not in excluded]
limits = previous['limits']

def fit(indices):
    rotations = [Rotation.from_matrix(hands[indices[0]][:3,:3].T @ hands[i][:3,:3]).as_rotvec() for i in indices[1:]]
    singular = np.linalg.svd(rotations, compute_uv=False)
    if singular[1] < .25 or singular[1] / max(singular[0], 1e-12) < .1:
        raise ValueError('Insufficient rotation diversity')
    inverses = [np.linalg.inv(hands[i]) for i in indices]
    r,t = cv2.calibrateHandEye([h[:3,:3] for h in inverses], [h[:3,3] for h in inverses],
         [markers[i][:3,:3] for i in indices], [markers[i][:3,3] for i in indices], method=cv2.CALIB_HAND_EYE_PARK)
    camera = checked_pose(pose(r,t))
    attached = [np.linalg.inv(hands[i]) @ camera @ markers[i] for i in indices]
    marker = pose(Rotation.from_matrix(np.array([a[:3,:3] for a in attached])).mean().as_matrix(), np.mean([a[:3,3] for a in attached],axis=0))
    mount = camera @ np.linalg.inv(checked_pose(data['mount_camera']))
    def errors(group):
        v = np.array([difference(hands[i] @ marker, camera @ markers[i]) for i in group])
        return dict(count=len(group), rms_position_m=float(np.sqrt(np.mean(v[:,0]**2))), max_position_m=float(v[:,0].max()), max_angle_deg=float(v[:,1].max()))
    training, check = errors(indices), errors(validation)
    return dict(accepted=all(e['max_position_m'] <= limits['max_position_m'] and e['max_angle_deg'] <= limits['max_angle_deg'] for e in (training,check)),
        base_camera=camera.tolist(), hand_marker=marker.tolist(), base_mount=mount.tolist(),
        chest_camera_xyz=mount[:3,3].tolist(), chest_camera_rpy=Rotation.from_matrix(mount[:3,:3]).as_euler('xyz').tolist(),
        training=training, validation=check, limits=limits)

# Verify this explicit split calculation reproduces the original solver first.
baseline = fit(original_train)
for key in ('base_camera', 'hand_marker', 'base_mount'):
    assert np.allclose(baseline[key], previous[key], atol=1e-9), key
for group in ('training', 'validation'):
    for key in ('rms_position_m','max_position_m','max_angle_deg'):
        assert np.isclose(baseline[group][key],previous[group][key],atol=1e-9), (group,key)
result = fit(train)
kept = [i for i in range(31) if i not in excluded]
result.update(parameters=data['parameters'], sample_count=len(kept), source_session=str(source),
    excluded_original_sample_numbers=[13,29], training_original_sample_numbers=[i+1 for i in train],
    validation_original_sample_numbers=[i+1 for i in validation],
    change_from_urdf_m_deg=difference(np.array(data['original_base_mount']),np.array(result['base_mount'])),
    note='Offline recalculation with explicit original holdout indices; use recalculate.py to reproduce, not default every-fifth splitting.')
filtered=dict(data, samples=[dict(samples[i],original_sample_number=i+1) for i in kept],
    excluded_original_sample_numbers=[13,29], validation_original_sample_numbers=[i+1 for i in validation],
    source_session=str(source))
write_json(folder/'samples.json', filtered)
write_json(folder/'result.json', result)
if result['accepted']:
    (folder/'chest_camera_origin.xacro.txt').write_text('\n'.join(f'<xacro:arg name="{key}" default="'+' '.join(f'{v:.9f}' for v in result[key])+'" />' for key in ('chest_camera_xyz','chest_camera_rpy'))+'\n')
print('saved',folder)
print('accepted',result['accepted'])
for key in ('training','validation'):
    e=result[key]
    print(key,'count=%d rms_mm=%.3f max_mm=%.3f max_deg=%.3f'%(e['count'],e['rms_position_m']*1000,e['max_position_m']*1000,e['max_angle_deg']))
print('holdout_original_numbers',result['validation_original_sample_numbers'])
