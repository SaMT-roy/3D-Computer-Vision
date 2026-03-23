import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from utils import *
import config as cfg

param_pack = ParamsPack()

def parse_param_62(param):
	"""
	Parses a 62-dimensional parameter tensor into its 3DMM components.

	Parameters
	----------
	param : torch.Tensor
		Tensor of shape (N, 62) containing model parameters, where:
			- first 12 values: camera matrix (3x4)
			- next 40 values: shape coefficients
			- last 10 values: expression coefficients

	Returns
	-------
	p : torch.Tensor
		Camera matrix (N, 3, 3).
	offset : torch.Tensor
		Translation vector (N, 3, 1).
	alpha_shp : torch.Tensor
		Shape coefficients (N, 40, 1).
	alpha_exp : torch.Tensor
		Expression coefficients (N, 10, 1).
	"""
	p_ = param[:, :12].reshape(-1, 3, 4)
	p = p_[:, :, :3]
	offset = p_[:, :, -1].reshape(-1, 3, 1)
	alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
	alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
	return p, offset, alpha_shp, alpha_exp

class featureclassifier(nn.Module):
    """
    Feature-based regressor for 3D Morphable Model (3DMM) parameters.

    This network uses a pretrained backbone (from `timm`) to extract deep
    visual features from face images and predicts three sets of parameters:
    orientation, shape, and expression.

    Parameters
    ----------
    base_model_name : str, optional
        Name of the pretrained backbone model from the `timm` library.
        Default is 'mobilevitv2_075.cvnets_in1k'.

    Attributes
    ----------
    base_model : nn.Module
        Pretrained CNN backbone providing feature maps.
    transforms : torchvision.transforms.Compose
        Preprocessing pipeline (resize, crop, normalization) matching the
        backbone's data configuration.
    classifier_ori : nn.Sequential
        Linear head predicting 12 orientation parameters.
    classifier_shape : nn.Sequential
        Linear head predicting 40 shape coefficients.
    classifier_exp : nn.Sequential
        Linear head predicting 10 expression coefficients.
    last_channel : int
        Dimension of the final feature vector from the backbone.
    """

    def __init__(self, base_model_name=cfg.feature_extraction_model):
        super(featureclassifier, self).__init__()
        self.base_model = timm.create_model(base_model_name, pretrained=True, features_only=True)
        self.base_model.eval()
        self.data_config = timm.data.resolve_model_data_config(self.base_model)

        self.size = self.data_config['input_size'][1:]  # (H, W)
        self.mean = self.data_config['mean']
        self.std = self.data_config['std']

        self.transforms = T.Compose([
            T.Resize(self.size),
            T.CenterCrop(self.size),
            T.Normalize(mean=self.mean, std=self.std)
        ])

        self.num_ori = 12
        self.num_shape = 40
        self.num_exp = 10
        self.last_channel = 384

        self.classifier_ori = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_ori),
        )
        self.classifier_shape = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_shape),
        )
        self.classifier_exp = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, self.num_exp),
        )

    def forward(self, x):
        """
        Forward pass through the network.

        Parameters
        ----------
        x : torch.Tensor
            Input image batch of shape (N, C, H, W).

        Returns
        -------
        x_3dmm : torch.Tensor
            Concatenated prediction vector containing orientation (12),
            shape (40), and expression (10) parameters. Shape: (N, 62).
        base_features : torch.Tensor
            Global feature embedding extracted from the backbone.
        """
        base_features = self.base_model(self.transforms(x))[-1]
        base_features = nn.functional.adaptive_avg_pool2d(base_features, 1)
        base_features = base_features.reshape(base_features.shape[0], -1)

        x_ori = self.classifier_ori(base_features)
        x_shape = self.classifier_shape(base_features)
        x_exp = self.classifier_exp(base_features)

        x_3dmm = torch.cat((x_ori, x_shape, x_exp), dim=1)
        return x_3dmm, base_features

class I2P(nn.Module):
	"""
	Image-to-Parameter (I2P) network that predicts 3D Morphable Model (3DMM)
	parameters from input face images.
	"""
	def __init__(self):
		super(I2P, self).__init__()
		self.backbone = featureclassifier()

	def forward(self, input, target):
		"""
		Forward pass during training.

		Parameters
		----------
		input : torch.Tensor
			Batch of input face images.
		target : torch.Tensor
			Ground-truth 3DMM parameter tensor.

		Returns
		-------
		_3D_attr : torch.Tensor
			Predicted 3DMM parameters.
		_3D_attr_GT : torch.Tensor
			Ground-truth parameters (same as `target`).
		avgpool : torch.Tensor
			Feature embedding from the backbone network.
		"""
		_3D_attr, avgpool = self.backbone(input)
		_3D_attr_GT = target
		return _3D_attr, _3D_attr_GT, avgpool

	def forward_test(self, input):
		"""
		Forward pass during inference (testing).

		Parameters
		----------
		input : torch.Tensor
			Batch of input face images.

		Returns
		-------
		_3D_attr : torch.Tensor
			Predicted 3DMM parameters.
		avgpool : torch.Tensor
			Feature embedding from the backbone network.
		"""
		_3D_attr, avgpool = self.backbone(input)
		return _3D_attr, avgpool

class MLP_for(nn.Module):
    """
    Multi-Layer Perceptron (MLP) network for point-wise feature learning and 3D reconstruction.

    This architecture is designed to process 3D point cloud data and integrate additional 
    global and semantic features (e.g., image features, shape, and expression codes) 
    to predict refined 3D vertex positions or deformations.

    **Core Idea:**
    The network combines:
        - Point-level geometric features (local spatial features)
        - Global context features (via pooling)
        - External conditioning inputs (image-derived features, shape and expression codes)
    
    The design is heavily inspired by *PointNet*, where point-wise MLPs and 
    global feature pooling are used to learn permutation-invariant representations 
    of unordered 3D points.

    Args:
        num_pts (int): Number of input points in the point cloud.

    Input:
        x: torch.Tensor of shape (B, 3, N)
            - B: Batch size
            - 3: Coordinates (x, y, z) for each point
            - N: Number of points (num_pts)
        
        other_input1: torch.Tensor of shape (B, 384)
            - Global image embedding (from backbone's avgpool output)

        other_input2: torch.Tensor of shape (B, 40)
            - Shape code (3DMM shape coefficients)
        
        other_input3: torch.Tensor of shape (B, 10)
            - Expression code (3DMM expression coefficients)

    Output:
        torch.Tensor of shape (B, 3, N)
            - Predicted refined 3D vertex positions per input point.

    Architecture Overview:
    ----------------------------------------------------------
    Input (B, 3, N)
        │
        ├── Conv1d(3 → 64) + BN + ReLU        # local feature extraction
        ├── Conv1d(64 → 64) + BN + ReLU       # refine local features
        ├── Conv1d(64 → 64 → 128 → 1024)      # deeper layers build global representation
        │
        ├── MaxPool(1024, N) → (B, 1024, 1)   # global context
        │
        ├── Concatenate point_features + global_features + avgpool + shape + expr
        │
        ├── Conv1d(1522 → 512 → 256 → 128 → 3)
        │
        └── Output: (B, 3, N)
    """

    def __init__(self, num_pts):
        super(MLP_for, self).__init__()

        # ----- Local Feature Extractors -----
        # Extract per-point geometric features progressively
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # ----- Feature Fusion & Reconstruction -----
        # Combine local + global + conditioning (image + shape + expression)
        self.conv6 = nn.Conv1d(1522, 512, 1)  # 64(local) + 1024(global) + 384(avgpool) + 40(shape) + 10(expr)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.conv8 = nn.Conv1d(256, 128, 1)
        self.conv9 = nn.Conv1d(128, 3, 1)

        # ----- Batch Normalization Layers -----
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6 = nn.BatchNorm1d(512)
        self.bn7 = nn.BatchNorm1d(256)
        self.bn8 = nn.BatchNorm1d(128)
        self.bn9 = nn.BatchNorm1d(3)

        self.num_pts = num_pts
        self.max_pool = nn.MaxPool1d(num_pts)

    def forward(self, x, other_input1=None, other_input2=None, other_input3=None):
        """
        Forward pass of the MLP network.

        Combines local geometric features with global and high-level semantic embeddings
        to reconstruct or refine 3D vertex positions for each point in the input cloud.

        Steps:
        1. Extract local point-wise features using a series of Conv1D + BN + ReLU.
        2. Aggregate global context using max-pooling.
        3. Broadcast global features to each point.
        4. Concatenate local, global, and auxiliary features (avgpool, shape, expr).
        5. Decode through MLP layers to predict (x, y, z) coordinates for each point.

        Args:
            x (Tensor): Input point cloud of shape (B, 3, N).
            other_input1 (Tensor): Global image features (B, 384).
            other_input2 (Tensor): Shape code (B, 40).
            other_input3 (Tensor): Expression code (B, 10).

        Returns:
            Tensor: Predicted 3D vertices of shape (B, 3, N).
        """

        # --- Step 1: Local Feature Extraction ---
        out = F.relu(self.bn1(self.conv1(x)))     # (B, 64, N)
        out = F.relu(self.bn2(self.conv2(out)))   # (B, 64, N)
        point_features = out                      # Save intermediate local features

        out = F.relu(self.bn3(self.conv3(out)))   # (B, 64, N)
        out = F.relu(self.bn4(self.conv4(out)))   # (B, 128, N)
        out = F.relu(self.bn5(self.conv5(out)))   # (B, 1024, N)

        # --- Step 2: Global Feature Aggregation ---
        global_features = self.max_pool(out)      # (B, 1024, 1)
        global_features_repeated = global_features.repeat(1, 1, self.num_pts)  # Broadcast to each point

        # --- Step 3: External Feature Conditioning ---
        avgpool = other_input1.unsqueeze(2).repeat(1, 1, self.num_pts)  # (B, 384, N)
        shape_code = other_input2.unsqueeze(2).repeat(1, 1, self.num_pts)  # (B, 40, N)
        expr_code = other_input3.unsqueeze(2).repeat(1, 1, self.num_pts)   # (B, 10, N)

        # --- Step 4: Feature Fusion ---
        # Combine all information channels
        fused = torch.cat(
            [point_features, global_features_repeated, avgpool, shape_code, expr_code],
            dim=1
        )

        # --- Step 5: Decoding & Output ---
        out = F.relu(self.bn6(self.conv6(fused)))
        out = F.relu(self.bn7(self.conv7(out)))
        out = F.relu(self.bn8(self.conv8(out)))
        out = F.relu(self.bn9(self.conv9(out)))  # Final output (B, 3, N)
        return out

class MLP_rev(nn.Module):
    """
    Multi-Layer Perceptron (MLP) network for regressing 3D Morphable Model (3DMM) parameters 
    — rotation, shape, and expression — from 3D point cloud features.

    This network performs the **reverse mapping** of `MLP_for`.  
    Instead of generating 3D vertices from 3DMM parameters, it learns to infer 
    those 3DMM parameters directly from 3D point-level geometry (e.g., facial mesh or landmarks).

    **Core Idea:**
    The input is a 3D point cloud representing a reconstructed face (or similar object).
    The network:
      1. Extracts local geometric features using per-point convolutions (Conv1D + BN + ReLU)
      2. Aggregates them into a single **global feature** using max pooling
      3. Decodes this global representation into:
         - 12D rotation/pose parameters
         - 40D shape coefficients
         - 10D expression coefficients
      These correspond to the same dimensional splits as the `I2P` or 3DMM structure.

    Args:
        num_pts (int): Number of input points in the point cloud.

    Input:
        x: torch.Tensor of shape (B, 3, N)
            - B: Batch size
            - 3: (x, y, z) coordinates per point
            - N: Number of points (num_pts)
        
        other_input1, other_input2, other_input3:
            - Reserved for compatibility (not used in this model)

    Output:
        torch.Tensor of shape (B, 62)
            - Concatenation of rotation (12), shape (40), and expression (10) parameters.
            - These form a 62-dimensional vector similar to `parse_param_62()` output.

    Architecture Overview:
    ----------------------------------------------------------
    Input (B, 3, N)
        │
        ├── Conv1d(3 → 64) + BN + ReLU        # local geometric features
        ├── Conv1d(64 → 64) + BN + ReLU       # refine point-level features
        ├── Conv1d(64 → 64 → 128 → 1024)      # deeper hierarchical features
        │
        ├── MaxPool1d(N) → (B, 1024, 1)       # global feature aggregation
        │
        ├── Conv1d(1024 → 12)  → rotation
        ├── Conv1d(1024 → 40)  → shape
        ├── Conv1d(1024 → 10)  → expression
        │
        └── Concatenate → (B, 62)
    """

    def __init__(self, num_pts):
        super(MLP_rev, self).__init__()

        # ----- Local feature extraction -----
        # Learn per-point features from raw 3D coordinates.
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, 1024, 1)

        # ----- Regression heads -----
        # Each branch predicts a specific 3DMM parameter group.
        self.conv6_1 = nn.Conv1d(1024, 12, 1)  # Rotation / Pose parameters
        self.conv6_2 = nn.Conv1d(1024, 40, 1)  # Shape coefficients (identity)
        self.conv6_3 = nn.Conv1d(1024, 10, 1)  # Expression coefficients

        # ----- Batch Normalization -----
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(1024)
        self.bn6_1 = nn.BatchNorm1d(12)
        self.bn6_2 = nn.BatchNorm1d(40)
        self.bn6_3 = nn.BatchNorm1d(10)

        self.num_pts = num_pts
        self.max_pool = nn.MaxPool1d(num_pts)

    def forward(self, x, other_input1=None, other_input2=None, other_input3=None):
        """
        Forward pass of the reverse MLP.

        This function encodes the 3D point cloud into a compact global descriptor 
        and decodes it into 3DMM parameter predictions (rotation, shape, expression).

        Args:
            x (Tensor): Input point cloud (B, 3, N)
            other_input1, other_input2, other_input3: (unused)

        Returns:
            Tensor: 3DMM parameter vector (B, 62)
        """

        # --- Step 1: Local feature extraction ---
        out = F.relu(self.bn1(self.conv1(x)))   # (B, 64, N)
        out = F.relu(self.bn2(self.conv2(out))) # (B, 64, N)
        out = F.relu(self.bn3(self.conv3(out))) # (B, 64, N)
        out = F.relu(self.bn4(self.conv4(out))) # (B, 128, N)
        out = F.relu(self.bn5(self.conv5(out))) # (B, 1024, N)

        # --- Step 2: Global feature aggregation ---
        global_features = self.max_pool(out)    # (B, 1024, 1)

        # --- Step 3: Decode into 3DMM parameter groups ---
        out_rot = F.relu(self.bn6_1(self.conv6_1(global_features)))   # (B, 12, 1)
        out_shape = F.relu(self.bn6_2(self.conv6_2(global_features))) # (B, 40, 1)
        out_expr = F.relu(self.bn6_3(self.conv6_3(global_features)))  # (B, 10, 1)

        # --- Step 4: Concatenate outputs ---
        out = torch.cat([out_rot, out_shape, out_expr], dim=1).squeeze(2)  # (B, 62)
        return out
	
# Main model SynergyNet definition
class SynergyNet(nn.Module):
    """
    SynergyNet is the main PyTorch module for a deep learning-based 3D face reconstruction model.

    This model implements a synergistic, dual-pathway approach to regress 3D Morphable Model (3DMM)
    parameters from a single 2D image. It consists of three main components:
    1.  I2P (Image-to-Parameter): A backbone network that predicts initial 3DMM parameters from an input image.
    2.  forwardDirection: An MLP-based network that refines the 3D landmark positions predicted from the
        initial parameters. This is the "forward" path.
    3.  reverseDirection: An MLP-based network that predicts 3DMM parameters from the refined 3D landmarks.
        This is the "reverse" path, creating a closed loop that allows for self-supervised consistency checks.

    The model is trained with a combination of losses on the 3D landmarks and the 3DMM parameters to ensure
    accurate and coherent reconstructions.

    Attributes:
        I2P (nn.Module): The Image-to-Parameter regression network.
        forwardDirection (nn.Module): The forward path MLP for refining landmarks.
        reverseDirection (nn.Module): The reverse path MLP for re-predicting parameters.
        LMKLoss_3D (nn.Module): The WingLoss function for 3D landmark alignment.
        ParamLoss (nn.Module): The L2 loss function for 3DMM parameter regression.
        loss (dict): A dictionary to store the values of different loss components during training.
        param_mean (torch.Tensor): The mean values for the 62 3DMM parameters, used for de-whitening.
        param_std (torch.Tensor): The standard deviation values for the 62 3DMM parameters.
        w_shp (torch.Tensor): The shape basis vectors of the 3DMM for dense mesh reconstruction.
        u (torch.Tensor): The mean shape vector of the 3DMM for the dense mesh.
        w_exp (torch.Tensor): The expression basis vectors of the 3DMM for dense mesh reconstruction.
        u_base (torch.Tensor): The mean shape vector of the 3DMM for sparse landmarks.
        w_shp_base (torch.Tensor): The shape basis vectors for sparse landmarks.
        w_exp_base (torch.Tensor): The expression basis vectors for sparse landmarks.
        keypoints (torch.Tensor): Indices mapping the dense mesh vertices to the 68 sparse landmarks.
    """
    def __init__(self):
        """Initializes all sub-modules, loss functions, and registers 3DMM constants as buffers."""
        super(SynergyNet, self).__init__()
        # Image-to-parameter network predicts initial 3DMM parameters from the input image.
        self.I2P = I2P()
        # Forward network refines 3D landmarks based on initial parameters and image features.
        self.forwardDirection = MLP_for(68)
        # Reverse network predicts 3DMM parameters from the refined 3D landmarks.
        self.reverseDirection = MLP_rev(68)
        
        # Loss functions for landmarks and parameters.
        self.LMKLoss_3D = WingLoss()
        self.ParamLoss = ParamLoss()
		
        # Dictionary to store and track individual loss components.
        self.loss = {'loss_LMK_f0':0.0,
					'loss_LMK_pointNet': 0.0,
					'loss_Param_In':0.0,
					'loss_Param_S2': 0.0,
					'loss_Param_S1S2': 0.0,
					}
        
        # Registering 3DMM statistical model parameters as buffers.
        # These are pre-computed constants and are not updated during training.
        self.register_buffer('param_mean', torch.tensor(param_pack.param_mean, dtype=torch.float32))
        self.register_buffer('param_std', torch.tensor(param_pack.param_std, dtype=torch.float32))
        self.register_buffer('w_shp', torch.tensor(param_pack.w_shp, dtype=torch.float32))
        self.register_buffer('u', torch.tensor(param_pack.u, dtype=torch.float32))
        self.register_buffer('w_exp', torch.tensor(param_pack.w_exp, dtype=torch.float32))
        # Buffers for the sparse (68 landmarks) model.
        self.register_buffer('u_base', torch.tensor(param_pack.u_base, dtype=torch.float32))
        self.register_buffer('w_shp_base', torch.tensor(param_pack.w_shp_base, dtype=torch.float32))
        self.register_buffer('w_exp_base', torch.tensor(param_pack.w_exp_base, dtype=torch.float32))
        # Indices for the 68 landmarks.
        self.keypoints = torch.tensor(param_pack.keypoints, dtype=torch.long)
		
        # Grouping some data parameters for convenience (not used in the provided snippet).
        self.data_param = [self.param_mean, self.param_std, self.w_shp_base, self.u_base, self.w_exp_base]
		
    def reconstruct_vertex_62(self, param, whitening=True, dense=False, transform=True, lmk_pts=68):
        """
        Reconstructs 3D vertices from a 62-dimensional 3DMM parameter vector.

        This function applies the linear 3DMM formulation to generate 3D facial vertices.
        It can produce either a dense mesh or a set of sparse landmarks.

        Args:
            param (torch.Tensor): A batch of 62-dimensional 3DMM parameters. Shape: (batch_size, 62).
            whitening (bool, optional): If True, de-normalize the parameters using pre-registered mean and
                                        std. Defaults to True.
            dense (bool, optional): If True, returns the dense 53215-vertex mesh. Otherwise, returns
                                    sparse landmarks. Defaults to False.
            transform (bool, optional): If True, transforms the vertices to the image coordinate space
                                        (y-axis flipped). Defaults to True.
            lmk_pts (int, optional): The number of sparse landmarks to reconstruct (e.g., 68).
                                     Defaults to 68.

        Returns:
            torch.Tensor: The reconstructed 3D vertices. Shape: (batch_size, 3, num_vertices).
        """
        
        # De-normalize parameters from a standard normal distribution to their original scale.
        if whitening:
            if param.shape[1] == 62:
                param_ = param * self.param_std[:62] + self.param_mean[:62]
            else:
                raise RuntimeError('length of params mismatch')
        else:
            param_ = param
            
        # Parse the 62-dim vector into its components: rotation, offset, shape alphas, and expression alphas.
        p, offset, alpha_shp, alpha_exp = parse_param_62(param_)

        if dense:
            # Reconstruct the dense mesh using the 3DMM formula: V = R * (U + W_shp*alpha_shp + W_exp*alpha_exp) + t
            vertex = p @ (self.u + self.w_shp @ alpha_shp + self.w_exp @ alpha_exp).contiguous().view(-1, 53215, 3).transpose(1,2) + offset
            
            if transform: 
                # Transform to image coordinate space by flipping the y-axis.
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        else:
            # Reconstruct sparse landmarks using the sparse 3DMM bases.
            vertex = p @ (self.u_base + self.w_shp_base @ alpha_shp + self.w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1,2) + offset

            if transform: 
                # Transform to image coordinate space.
                vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

        return vertex
	
    def forward(self, input, target):
        """
        Defines the forward pass of the model during training.

        This method processes an input image and its corresponding ground truth 3DMM parameters to compute
        a set of losses that drive the training process.

        Args:
            input (torch.Tensor): A batch of input images.
            target (torch.Tensor): A batch of ground truth 62-dimensional 3DMM parameters.

        Returns:
            dict: A dictionary containing the computed loss values.
        """
        # Step 1: Initial parameter prediction from the image.
        _3D_attr, _3D_attr_GT, avgpool = self.I2P(input, target)
        
        # Reconstruct 3D landmarks from both predicted and ground truth parameters.
        vertex_lmk = self.reconstruct_vertex_62(_3D_attr, dense=False)
        vertex_GT_lmk = self.reconstruct_vertex_62(_3D_attr_GT, dense=False)
		
        # Calculate initial losses based on the direct output of the I2P network.
        # Loss 1: Landmark loss between initially reconstructed landmarks and ground truth landmarks.
        self.loss['loss_LMK_f0'] = 0.05 * self.LMKLoss_3D(vertex_lmk, vertex_GT_lmk, kp=True)
        # Loss 2: Parameter loss between predicted parameters and ground truth parameters.
        self.loss['loss_Param_In'] = 0.02 * self.ParamLoss(_3D_attr, _3D_attr_GT)

        # Step 2: Forward pass - Refine landmark positions.
        # The MLP takes initial landmarks, image features, and parameter subsets to predict a residual.
        point_residual = self.forwardDirection(vertex_lmk, avgpool, _3D_attr[:,12:52], _3D_attr[:,52:62])
        # Add the predicted residual to refine the landmarks.
        vertex_lmk_refined = vertex_lmk + 0.05 * point_residual
		
        # Loss 3: Landmark loss on the *refined* landmarks.
        self.loss['loss_LMK_pointNet'] = 0.05 * self.LMKLoss_3D(vertex_lmk_refined, vertex_GT_lmk, kp=True)

        # Step 3: Reverse pass - Predict parameters from the refined landmarks.
        _3D_attr_S2 = self.reverseDirection(vertex_lmk_refined)
		
        # Loss 4: Parameter loss between the re-predicted parameters and the ground truth.
        self.loss['loss_Param_S2'] = 0.02 * self.ParamLoss(_3D_attr_S2, _3D_attr_GT, mode='only_3dmm')
        # Loss 5: Consistency loss between the initial parameters and the re-predicted parameters.
        self.loss['loss_Param_S1S2'] = 0.001 * self.ParamLoss(_3D_attr_S2, _3D_attr, mode='only_3dmm')
		
        return self.loss

    def forward_test(self, input):
        """
        Defines the forward pass of the model for inference/testing.

        This method takes an input image and returns the predicted 3DMM parameters without
        calculating any losses.

        Args:
            input (torch.Tensor): A batch of input images.

        Returns:
            torch.Tensor: The predicted 62-dimensional 3DMM parameters.
        """
        # Predict 3DMM attributes directly from the image using the I2P network in test mode.
        _3D_attr, _ = self.I2P.forward_test(input)
        return _3D_attr

    def get_losses(self):
        """
        Returns the names of the losses tracked by the model.

        Returns:
            dict_keys: An object containing the keys (names) of the loss dictionary.
        """
        return self.loss.keys()