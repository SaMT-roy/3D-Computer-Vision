import torch
import torch.nn as nn
import torch.utils.data as data
import math
import pickle
import os.path as osp
import numpy as np
import cv2
from pathlib import Path

class WingLoss(nn.Module):
    """
    Wing Loss implementation for regression tasks, particularly effective for landmark detection
    (e.g., facial keypoints) where small errors need strong penalization for precision, while
    being robust to large outliers via a linear regime.

    The Wing Loss function is defined piecewise:
    - For small errors (δ = |y - ŷ| < ω): L = ω * log(1 + δ / ε)
      This logarithmic term amplifies gradients for fine-grained adjustments, improving convergence
      on subtle displacements (e.g., pixel-level accuracy in 3D landmarks).

    - For large errors (δ ≥ ω): L = δ - C
      This linear term (similar to L1) prevents gradient explosion from outliers, promoting stability.
      
    - C = ω - ω * log(1 + ω / ε) ensures continuity at δ = ω.

    The loss is computed coordinate-wise (x, y, z for each landmark) across the batch, then averaged
    over all elements (mean loss). This is suitable for 3D vertex regression in models like 3DMM.

    Args:
        omega (float, optional): Threshold for switching from log to linear regime (default: 10).
            Controls sensitivity to small vs. large errors; higher values emphasize small errors more.
        epsilon (float, optional): Small constant to avoid log(0) (default: 2). Stabilizes gradients
            near zero errors.
    """
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon
        self.log_term = math.log(1 + self.omega / self.epsilon)

    def forward(self, pred, target, kp=False):
        # pred has shape [batch_size, 3, n_points], this extracts how many 3D points (landmarks) are in each sample.
        n_points = pred.shape[2]

        # Flatten all landmarks into one long vector of coordinates (x, y, z for each landmark).
        pred = pred.transpose(1, 2).contiguous().view(-1, 3 * n_points)
        target = target.transpose(1, 2).contiguous().view(-1, 3 * n_points)

        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()

        # Logarithmic - Penalizes small errors more strongly than L1 — improves fine-grained accuracy (e.g., precise landmark localization)
        delta_y1 = delta_y[delta_y < self.omega]
        # Linear - Like L1, so it’s robust to outliers — doesn’t explode with large deviations.
        delta_y2 = delta_y[delta_y >= self.omega]

        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * self.log_term
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))

class ParamLoss(nn.Module):
    """
    Custom loss module for regressing 62-dimensional parameters in 3D Morphable Model (3DMM)
    fitting tasks, such as face reconstruction. Computes root mean squared error (RMSE) variants
    on pose (first 12 elements: 3x4 projection matrix flattened) and 3DMM parameters (next 50:
    40 shape + 10 expression coefficients).

    Expects input and target tensors of shape [batch_size, 62], where each row is a flattened
    parameter vector. Uses nn.MSELoss with reduction='none' internally, then applies mean(dim=1)
    per sample and sqrt for RMSE. Supports modes to weight pose vs. 3DMM components differently.

    Modes:
    - 'normal': Separate RMSE on pose (input[:, :12] vs target[:, :12]) + RMSE on 3DMM params
      (input[:, 12:] vs target[:, 12:]), summed per sample. Balances pose accuracy with shape/expression.
    - 'only_3dmm': RMSE on full 3DMM params (input[:, :50] vs target[:, 12:62]), ignoring pose
      in input but using target's 3DMM slice. Useful for focusing on shape/expression refinement.
    - Default (any other mode): RMSE over entire 62D vector (input vs target).

    This loss promotes stable training by treating pose and deformable params separately, common
    in cascaded 3DMM pipelines where pose is estimated first.
    """
    def __init__(self):
        super(ParamLoss, self).__init__()
        self.criterion = nn.MSELoss(reduction="none")

    def forward(self, input, target, mode='normal'):
        """
        Compute parameterized RMSE loss between predicted and target parameters.

        Args:
            input (torch.Tensor): Predicted parameters [batch_size, 62].
            target (torch.Tensor): Ground-truth parameters [batch_size, 62].
            mode (str, optional): Loss mode ('normal', 'only_3dmm', or default) (default: 'normal').

        Returns:
            torch.Tensor: Per-sample RMSE losses [batch_size].

        Raises:
            ValueError: If input/target shapes mismatch (must be [B, 62]).
        """
        if input.shape != target.shape or input.shape[1] != 62:
            raise ValueError("Input and target must be tensors of shape [batch_size, 62].")
        
        if mode == 'normal':
            loss = self.criterion(input[:, :12], target[:, :12]).mean(1) + \
                   self.criterion(input[:, 12:], target[:, 12:]).mean(1)
            return torch.sqrt(loss)
        elif mode == 'only_3dmm':
            loss = self.criterion(input[:, :50], target[:, 12:62]).mean(1)
            return torch.sqrt(loss)
        # Default: full parameter RMSE
        loss = self.criterion(input, target).mean(1)
        return torch.sqrt(loss)

class ParamsPack():
    """
    Parameter package for a 3D Morphable Model (3DMM), a statistical model of human faces
    that generates diverse face shapes and expressions by linearly combining principal
    components from a mean face template.

    A 3DMM represents faces as a linear subspace:
    - The "Base Head" is the Mean Face (u).
    - The "Sliders" are the Principal Components (w_shp, w_exp). They define the fundamental rules of how a face can change.
    - The "Slider Values" are the Parameters (alpha_shp, alpha_exp).

    Key components:
    1. w_shp: Shape Basis
       Each column of this matrix is a "direction" for changing the face's identity.
       The first column might represent the direction for "long face vs. round face."
       The second column might be "wide jaw vs. narrow jaw."
       These were learned using Principal Component Analysis (PCA) on thousands of 3D face scans.
       Shape: (3 * N_vertices, 40), where N_vertices is the number of mesh vertices (e.g., ~50k for dense).

    2. w_exp: Expression Basis.
       Similarly, each column is a direction for changing the facial expression, like "smile," "frown," or "mouth open."
       Shape: (3 * N_vertices, 10).

    3. param_mean and param_std:
       "Whitening" is a data normalization technique.
       When a model (e.g., SynergyNet) predicts the "slider values," it predicts them in a normalized (whitened) space.
       These param_mean and param_std values are used to denormalize (un-whiten) the predicted values back into a meaningful range.
       Shape: (62,) for full parameters (12 pose + 40 shape + 10 expression).

    4. u: Mean 3D Shape (Neutral Expression)
       A long vector of (x, y, z, x, y, z, ...) coordinates for all vertices of the default, neutral, average human face.
       This is the "Base Head" or the Average Face, serving as the starting point.
       All new faces are created by taking this mean face and adding modifications from the sliders.
       Computed as u = u_shp + u_exp (mean shape + neutral expression mean).
       Shape: (3 * N_vertices,).

    Additional attributes for base/landmark-only computations (e.g., for efficiency with 68 or 5328 keypoints):
    - u_base: Mean vertices for keypoints only. Shape: (3 * N_keypoints, 1).
    - w_shp_base, w_exp_base: Subsampled bases for keypoints. Shape: (3 * N_keypoints, 40/10).
    - keypoints: Indices of keypoint vertices in the full mesh.
    - w_base: Concatenated base basis (w_shp_base + w_exp_base).
    - w_norm, w_base_norm: L2 norms of basis vectors for optional normalization.
    - std_size: Standard normalized image size (120 pixels).
    - dim: Number of vertices in the full mesh (w_shp.shape[0] // 3).

    Usage:
        pp = ParamsPack()
        # Denormalize params: param_ = param * pp.param_std[:62] + pp.param_mean[:62]
        # Blend shape: blended = pp.u + pp.w_shp @ alpha_shp + pp.w_exp @ alpha_exp
        # For keypoints only: blended_base = pp.u_base + pp.w_shp_base @ alpha_shp + pp.w_exp_base @ alpha_exp

    Raises:
        RuntimeError: If required data files are missing in the specified directory.
    """
    def __init__(self):
        """
        Data directory: '/Users/saptarshimallikthakur/Pictures/VLM/Synergynet/3dmm_data'

        Loads:
        - keypoints_sim.npy: Keypoint indices.
        - w_shp_sim.npy, w_exp_sim.npy: PCA bases.
        - param_whitening.pkl: Normalization metadata.
        - u_shp.npy, u_exp.npy: Mean shape and expression.

        Computes derived attributes like u, w, base versions, norms, etc.
        """
        try:
            d = '/Users/saptarshimallikthakur/Pictures/VLM/Synergynet/3dmm_data'
            self.keypoints = _load(osp.join(d, 'keypoints_sim.npy'))

            # PCA basis for shape, expression, texture
            self.w_shp = _load(osp.join(d, 'w_shp_sim.npy'))
            self.w_exp = _load(osp.join(d, 'w_exp_sim.npy'))
            # param_mean and param_std are used for re-whitening
            meta = _load(osp.join(d, 'param_whitening.pkl'))
            self.param_mean = meta.get('param_mean')
            self.param_std = meta.get('param_std')
            # mean values
            self.u_shp = _load(osp.join(d, 'u_shp.npy'))
            self.u_exp = _load(osp.join(d, 'u_exp.npy'))
            self.u = self.u_shp + self.u_exp
            self.w = np.concatenate((self.w_shp, self.w_exp), axis=1)
            # base vector for landmarks
            self.w_base = self.w[self.keypoints]
            self.w_norm = np.linalg.norm(self.w, axis=0)
            self.w_base_norm = np.linalg.norm(self.w_base, axis=0)
            self.u_base = self.u[self.keypoints].reshape(-1, 1)
            self.w_shp_base = self.w_shp[self.keypoints]
            self.w_exp_base = self.w_exp[self.keypoints]
            self.std_size = 120
            self.dim = self.w_shp.shape[0] // 3
        except:
            raise RuntimeError('Missing data')
        
def make_abs_path(d):
    return osp.join(osp.dirname(osp.realpath(__file__)), d)

def _get_suffix(filename):
    """a.jpg -> jpg"""
    pos = filename.rfind('.')
    if pos == -1:
        return ''
    return filename[pos + 1:]

def _load(fp):
    suffix = _get_suffix(fp)
    if suffix == 'npy':
        return np.load(fp)
    elif suffix == 'pkl':
        return pickle.load(open(fp, 'rb'))
          
class ToTensor(object):
    def __call__(self, pic):
        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            return img.float()

    def __repr__(self):
        return self.__class__.__name__ + '()'

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        tensor.sub_(self.mean).div_(self.std)
        return tensor

class Compose_GT(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

def img_loader(path):
    return cv2.imread(path, cv2.IMREAD_COLOR)

class DDFADataset(data.Dataset):
    def __init__(self, root, filelists, param_fp, transform=None, **kargs):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')
        self.params = _numpy_to_tensor(_load_cpu(param_fp))
        self.img_loader = img_loader

    def _target_loader(self, index):
        target_param = self.params[index]
        return target_param

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = self.img_loader(path)
        target = self._target_loader(index)
        img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.lines)
    
_numpy_to_tensor = lambda x: torch.from_numpy(x)
_load_cpu = _load

# __all__ = ['_load', '_numpy_to_tensor','_load_cpu']
