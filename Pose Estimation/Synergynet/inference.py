import numpy as np
import matplotlib.pyplot as plt
from utils import ParamsPack
from math import cos, sin, atan2, asin, sqrt
import cv2

param_pack = ParamsPack()

def parse_param(param):
    """
    Parse the input parameter vector into components for 3D model fitting,
    such as pose, translation, shape, and expression parameters.

    This function assumes the input `param` is a flattened NumPy array
    containing concatenated model parameters in the order:
    - Pose (rotation matrix elements): first 9 elements
    - Translation (offset): next 3 elements
    - Shape coefficients (alpha_shp): next 40 elements
    - Expression coefficients (alpha_exp): last 10 elements

    Args:
        param (numpy.ndarray): Input parameter vector of shape (62,).

    Returns:
        tuple:
            - p (numpy.ndarray): Rotation matrix of shape (3, 3).
            - offset (numpy.ndarray): Translation vector of shape (3, 1).
            - alpha_shp (numpy.ndarray): Shape parameters of shape (40, 1).
            - alpha_exp (numpy.ndarray): Expression parameters of shape (10, 1).
    """
    p_ = param[:12].reshape(3, 4)
    p = p_[:, :3]
    offset = p_[:, -1].reshape(3, 1)
    alpha_shp = param[12:52].reshape(40, 1)
    alpha_exp = param[52:62].reshape(10, 1)
    return p, offset, alpha_shp, alpha_exp

def P2sRt(P):
    """
    ------------------------------------------------------------------------
    P is camera projection matrix that encodes scale + rotation + translation all mixed together.
    Need to extract Scale (s), Rotation matrix (R), and Translation (t3d) from the 3x4 camera projection matrix P.
    ------------------------------------------------------------------------

    Given:
        P = [ s*R | t ]
            where:
                - s : scale factor
                - R : 3x3 rotation matrix
                - t : 3x1 translation vector

    Steps:
    1. t3d = P[:, 3] → last column is translation.
    2. R1, R2 → first two rows of P (containing scaled rotation vectors).
    3. Compute scale which is the average of their norms.:
           s = (||R1|| + ||R2||) / 2
    4. Normalize R1, R2 to get unit vectors:
           r1 = R1 / ||R1||,  r2 = R2 / ||R2||

        Because a true rotation matrix has unit-length rows (‖R1‖ = ‖R2‖ = 1).
        But when multiplied by scale s, they become sR1 and sR2.
        So dividing by their norm (average) gives back the true scale.

    5. Compute third axis via cross product:
           r3 = r1 × r2
    6. Reconstruct rotation matrix:
           R = [r1; r2; r3]

    Returns:
        s    : scalar (average scale)
        R    : 3x3 rotation matrix
        t3d  : 3x1 translation vector ([tx,ty,tz])

    ------------------------------------------------------------------------
    Example diagram:

         Camera matrix P
         ┌                         ┐
         │ s*r11  s*r12  s*r13  tx │  → R1 = first row
         │ s*r21  s*r22  s*r23  ty │  → R2 = second row
         │ s*r31  s*r32  s*r33  tz │
         └                         ┘
              ↓ normalize R1,R2
              ↓ compute r3 = r1×r2
              ↓ assemble rotation R

    Nuance Understanding:

    what is P really? 
    
    P (3×4) = [sR | t] represents the weak perspective camera projection used in 3DMM-based face reconstruction.

        ┌ x₂D ┐     ┌                ┐
        │     │  =  │ s · R₂×₃ · X₃D   + t₂D 
        └ y₂D ┘     └                ┘

    That means only the first two rows of the rotation matrix R affect the 2D projection (x and y coordinates).
    The third row of R affects depth (z) — but we don’t directly observe that in 2D images, so the model never really learns it reliably.
    Therefore, any values learned by a neural network for the third row of the sR sub-matrix are unconstrained by the 2D loss function and should be considered unreliable.

    So why take only first two rows?

        Because those are the only ones that: 
          the model predicts with any confidence, 
          and contain the actual projection information we can trust.

    Rotation matrix R:

    R = [ r₁      → direction of the rotated x-axis    
          r₂      → direction of the rotated y-axis
          r₃ ]    → direction of the rotated z-axis

    For a valid rotation matrix:

        - all three are orthogonal (perpendicular to each other)
        - each has unit length
        - det(𝑅) = 1

    Now, in the 3×4 affine matrix [sR | t], only the first two rows (R1 and R2) are reliably predicted by the model, 
    because 3D → 2D projection equations only use the first two rows.

    When you have two orthonormal vectors, r1 and r2, 
    you can compute the third orthogonal vector as their cross product:
                    
                    r3​=r1​×r2​

    That’s because:
    
        - The cross product of two vectors gives a vector perpendicular to both.
        - If r1 and r2 are unit-length and orthogonal, then r3 will also be unit-length and 
          orthogonal to both → forming a perfect right-handed coordinate system.

    The model only gives reliable info for R1 and R2 (since those are used in projecting 3D to image plane).
    We compute R3 manually so that:

        - the resulting rotation is orthonormal
        - the 3D object (like the face) doesn’t get skewed or sheared when rotated.

    Notes:

    Two vectors are orthogonal if their dot product is zero.
    i.e  a⋅b=0. This means they are perpendicular (at a 90° angle) to each other.

    Two (or more) vectors are orthonormal if they are:
        Orthogonal to each other (dot product = 0), and
        Each has unit length (magnitude = 1).

        ∥a∥ = ∥b∥= 1, a⋅b=0

    A set of vectors {v1,v2,v3,...} is an orthonormal basis if:
        Every pair is orthogonal,
        Each has unit length,
        Together they span the space.

    For example, in 3D:
        i = [1,0,0], j=[0,1,0], k=[0,0,1]
        These three form an orthonormal basis of R^3.

    ------------------------------------------------------------------------
    """
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def matrix2angle_corr(R):
    """
    Converts a 3×3 rotation matrix into Euler angles (in degrees),
    handling both normal and gimbal-lock cases.

    ---
    WHY / PURPOSE
    --------------
    The function extracts pitch (x), yaw (y), and roll (z) angles from a 3D rotation matrix `R`.  
    These angles describe how the head (or object) is rotated relative to the camera’s coordinate axes.

    ---
    SERIALISED EXPLANATION
    -----------------------
    1️⃣  The rotation matrix R encodes orientation in 3D as:
         R = Rz(roll) · Ry(yaw) · Rx(pitch)

        Each column (or row) of R represents an axis of the rotated coordinate frame
        expressed in camera coordinates.

    2️⃣  To get Euler angles back from R, we invert that relationship.
        The element R[2,0] = sin(pitch) gives us pitch (rotation about x-axis):
            x = asin(R[2,0])

    3️⃣  Once we know pitch (x), we can isolate yaw and roll using the remaining
        matrix entries.  
        Yaw (rotation about y-axis) and Roll (rotation about z-axis) are derived as:
            y = atan2(R[1,2]/cos(x), R[2,2]/cos(x))
            z = atan2(R[0,1]/cos(x), R[0,0]/cos(x))

        These formulas come from standard decomposition of a rotation matrix
        under the intrinsic rotation order (x → y → z).

    4️⃣  When R[2,0] = ±1, the pitch angle is ±90°, meaning the system loses one
        degree of rotational freedom (gimbal lock).  
        In that case, we manually set roll = 0 and solve yaw from the remaining entries.

    5️⃣  Finally, we convert the radians (x, y, z) into degrees and return:
            [rx, ry, rz] = [pitch, yaw, roll]

    ---
    RETURN
    -------
    A list [rx, ry, rz] in degrees:
      - rx → pitch (up-down head rotation)
      - ry → yaw   (left-right head rotation)
      - rz → roll  (tilt rotation)

    ---
    GEOMETRIC INTUITION
    -------------------
            ↑ y (up)
            |\
            | \
            |  \  ↺  pitch (x-axis rotation)
            |
            +------→ x (right)
           /
          /
         ↓ z (forward, camera view)

    As the head turns:
      • yaw → turning left/right (around y-axis)
      • pitch → nodding up/down (around x-axis)
      • roll → tilting side-to-side (around z-axis)
    """
    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))
    else:  # Gimbal lock case
        z = 0
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])

    rx, ry, rz = x * 180 / np.pi, y * 180 / np.pi, z * 180 / np.pi
    return [rx, ry, rz]

def parse_pose(param):
    """
    ------------------------------------------------------------------------
    Parses model parameters to extract pose (rotation, translation).
    ------------------------------------------------------------------------

    Steps:
    1. Normalize parameters back to real scale:
           param = param * std + mean
    2. Extract first 12 elements as 3x4 camera matrix:
           P = param[:12].reshape(3, 4)
    3. Decompose into s, R, t via P2sRt(P).
    4. Recombine rotation + translation (without scale):
           [R | t]
    5. Compute Euler angles from R.

    Returns:
        P     : 3x4 pose matrix [R | t]
        pose  : [yaw, pitch, roll] angles
        t3d   : translation vector
    ------------------------------------------------------------------------
    """
    if len(param)==62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        param = param * param_pack.param_std + param_pack.param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)
    pose = matrix2angle_corr(R)
    return P, pose, t3d

def predict_pose(param, roi_bbox=False, ret_mat=False):
    """
    Predict camera pose from a normalized parameter vector, optionally adjusting translation
    for a region-of-interest (ROI) bounding box and returning either the projection matrix
    or Euler angles with adjusted translation.

    This function wraps `parse_pose` to extract pose components and applies ROI-based scaling
    and translation to the 3D translation vector `t3d` (x, y components only) if `roi_bbox` is provided.
    The scaling assumes the model was trained/fitted on a 120x120 normalized crop (common in face models),
    so it rescales the translation to pixel coordinates within the ROI.

    Args:
        param (numpy.ndarray): 62D normalized parameter vector (first 12 for pose).
        roi_bbox (tuple or bool, optional): If True, use provided ROI; else False (default).
            Expected format: (sx, sy, ex, ey, confidence) where sx/sy/ex/ey are pixel coordinates
            (start_x, start_y, end_x, end_y). The fifth element (e.g., confidence score) is ignored.
        ret_mat (bool, optional): If True, return the 3x4 projection matrix P; else return angles and t3d (default False).

    Returns:
        If ret_mat=True:
            numpy.ndarray: 3x4 unscaled projection matrix P = [R | t3d].
        Else:
            tuple:
                - angles (list): Euler angles [yaw_deg, pitch_deg, roll_deg].
                - t3d (numpy.ndarray): Adjusted 3x1 translation vector.

    Raises:
        ValueError: If `param` shape is invalid, `roi_bbox` is malformed, or denormalization fails.

    Note:
        Requires global `param_pack` for denormalization (as in `parse_pose`).
        z-component of t3d is not scaled, assuming it's in a normalized depth space.
        This is common in 3D face reconstruction pipelines (e.g., adjusting head pose to image ROI).
    """
    P, angles, t3d = parse_pose(param)
    if roi_bbox is not None and len(roi_bbox) >= 4:
        sx, sy, ex, ey= roi_bbox
        scale_x = (ex - sx) / 120.0 # ROI width normalized to model input width
        scale_y = (ey - sy) / 120.0 # ROI height normalized to model input height
        t3d[0] = t3d[0] * scale_x + sx # Adjust x-translation to ROI x-start
        t3d[1] = t3d[1] * scale_y + sy # Adjust y-translation to ROI y-start
        # t3d[2] remains unchanged (z-depth in normalized space, or as-is if no scaling needed)
    if ret_mat:
        return P
    return angles, t3d

def param2vert(param, dense=False, transform=True):
    """
    Generate 3D vertices from a 62-dimensional parameter vector for a 3D Morphable Model (3DMM),
    such as a face model, by blending the mean shape with shape and expression coefficients,
    applying rotation and translation, and optionally transforming to image coordinate space.

    The function assumes a pre-trained 3DMM where:
    - `param_pack` is a global object containing model components:
      - `param_mean` and `param_std`: For denormalizing input parameters.
      - `u` or `u_base`: Mean vertex positions (3 x N_vertices).
      - `w_shp` or `w_shp_base`: Shape basis matrices (3*N_vertices x 40).
      - `w_exp` or `w_exp_base`: Expression basis matrices (3*N_vertices x 10).
      - `std_size`: Standard image size (e.g., 120 for normalized crop).
    - Input `param` is normalized; denormalization: param_ = param * std + mean.
    - For `dense=True`: Uses full dense mesh (e.g., ~50k vertices for detailed face).
    - For `dense=False`: Uses base/coarse mesh (e.g., 5328 vertices for landmarks).
    - Vertex computation: V = R @ (U + W_shp @ alpha_shp + W_exp @ alpha_exp) + t,
      where R=p (3x3 rotation), t=offset (3x1 translation), reshaped column-major (Fortran order).

    - If `transform=True`: Flips y-coordinate to image space: y' = std_size + 1 - y
      (converts from bottom-left origin, e.g., OpenGL, to top-left image origin).

      1. This is the standard system used in math, physics, and 3D modeling software.
            +Y (Up)
            |
            |
            o------+X (Right)
            /
            /
        +Z (Out of screen)

      2. The origin (0, 0) is at the center.
         The Y-axis increases as you go UP. A point at the top of the head has a large, positive Y value. A point at the chin has a negative Y value.

      3. This is the system used for images, where you count pixels from the corner.

            o--------> +X (Right)
            |
            |
            |
            +Y (Down)

            The origin (0, 0) is at the TOP-LEFT corner.
            The Y-axis increases as you go DOWN. A pixel at the top of the image has a small Y value (e.g., 10). 
            A pixel at the bottom has a large Y value (e.g., 470).

            The param2vert function creates a 3D face in the "3D Graphics World." 
            Before you can correctly project this face onto a "2D Image World," you must make their coordinate systems compatible.
            The line of code:
                              vertex[1, :] = param_pack.std_size + 1 - vertex[1, :] is the bridge.


    Args:
        param (numpy.ndarray): 62D normalized parameter vector.
            - [:12]: Pose (projection matrix elements).
            - [12:52]: Shape coefficients (40D).
            - [52:62]: Expression coefficients (10D).
        dense (bool, optional): If True, use dense/full mesh; else base/coarse mesh (default False).
        transform (bool, optional): If True, transform vertices to image coordinate space (default True).

    Returns:
        numpy.ndarray: 3 x N_vertices array of 3D vertices (x, y, z per column).

    Raises:
        RuntimeError: If `param` length != 62.
        ValueError: If shapes mismatch in matrix multiplications or `param_pack` is invalid.

    Note:
        Requires global `param_pack` with all necessary attributes.
        Output vertices are in world/camera space before ROI adjustment (use `_predict_vertices` for that).
        Fortran order ('F') ensures column-major reshaping for compatibility with mesh formats.
    """
    if param.shape[0] == 62:
        param_ = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        raise RuntimeError('length of params mismatch')

    p, offset, alpha_shp, alpha_exp = parse_param(param_)

    if dense:
        blended = param_pack.u + param_pack.w_shp @ alpha_shp + param_pack.w_exp @ alpha_exp
        vertex = p @ blended.reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]
    else:
        blended = param_pack.u_base + param_pack.w_shp_base @ alpha_shp + param_pack.w_exp_base @ alpha_exp
        vertex = p @ blended.reshape(3, -1, order='F') + offset
        if transform: 
            # transform to image coordinate space
            vertex[1, :] = param_pack.std_size + 1 - vertex[1, :]

    return vertex


def _predict_vertices(param, dense, roi_bbox=False, transform=True):
    """
    Predict and adjust 3D vertices to a region-of-interest (ROI) bounding box, building on `param2vert`
    by applying scaling and translation to map normalized vertices to pixel coordinates within the ROI.

    When a model like SynergyNet predicts 3D face vertices (vertex), they’re typically predicted in a normalized 
    coordinate space — for example, a fixed-size canonical space like 120×120.But the original input image 
    (where the face was detected) might have a different bounding box — the face could be larger, smaller, 
    or positioned elsewhere.

    So, to correctly overlay or visualize the predicted 3D landmarks/mesh on the original image, you need to:

        1. Scale the vertex coordinates to the bounding box size, and
        2. Translate them back to the bounding box’s position in the original image.

    This function first generates vertices using `param2vert`, then if `roi_bbox` is provided:

    - sx, sy → starting (top-left) coordinates of the face bounding box
    - ex, ey → ending (bottom-right) coordinates of the bounding box

    - Computes scales: scale_x = (ex - sx) / 120, scale_y = (ey - sy) / 120 (assuming model trained on 120x120 normalized crop).

      (ex - sx) and (ey - sy) represent how big the bounding box is in the original image.
      ratios tell you how much to scale the vertex x/y coordinates to map back.

    - Translates x,y: vertex[0,:] *= scale_x + sx; vertex[1,:] *= scale_y + sy.
    
        The x-coordinates are scaled and shifted by sx (left edge of the box).
        The y-coordinates are scaled and shifted by sy (top edge of the box).
        This moves vertices from normalized 120×120 space to the original image position.

    - Scales z-depth: s = (scale_x + scale_y)/2; vertex[2,:] *= s (isotropic scaling for depth to match average xy scale).

        The z-axis (depth) is scaled proportionally to the average of x and y scales,
        to maintain realistic proportions in 3D when the face size changes.

    Args:
        param (numpy.ndarray): 62D normalized parameter vector (as in `param2vert`).
        dense (bool): If True, use dense mesh; else base mesh (as in `param2vert`).
        roi_bbox (tuple or bool, optional): If provided, adjust to ROI; else False (default).
            Format: (sx, sy, ex, ey, confidence) pixel coordinates; fifth element ignored.
        transform (bool, optional): Passed to `param2vert` for image coord flip (default True).

    Returns:
        numpy.ndarray: 3 x N_vertices array of adjusted 3D vertices.

    Raises:
        ValueError: If `param` invalid, `roi_bbox` malformed, or scaling fails.
        RuntimeError: Propagated from `param2vert`.

    Note:
        Requires global `param_pack`.
        z-scaling uses average xy-scale for approximate orthographic projection consistency.
        The underscore prefix indicates internal/private use (e.g., in prediction pipelines).
        For pose-only, use `predict_pose`; this is for full vertex prediction with ROI.
    """
    vertex = param2vert(param, dense=dense, transform=transform)

    if roi_bbox is not None and len(roi_bbox) >= 4:
        sx, sy, ex, ey = roi_bbox
        scale_x = (ex - sx) / 120.0
        scale_y = (ey - sy) / 120.0
        vertex[0, :] = vertex[0, :] * scale_x + sx
        vertex[1, :] = vertex[1, :] * scale_y + sy

        s = (scale_x + scale_y) / 2.0
        vertex[2, :] *= s

    return vertex

def predict_sparseVert(param, roi_box, transform=False):
    return _predict_vertices(param, roi_bbox=roi_box, dense=False, transform=transform)

def predict_denseVert(param, roi_box, transform=False):
    return _predict_vertices(param, roi_bbox=roi_box, dense=True, transform=transform)

def draw_landmarks(img, pts, alpha = 1, markersize = 2, lw = 1.5, color='red'):
    height, width = img.shape[:2]
    base = 6.4 
    plt.figure(figsize=(base, height / width * base))
    plt.imshow(img[:,:,::-1])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.axis('off')

    if not type(pts) in [tuple, list]:
        pts = [pts]
    for i in range(len(pts)):
        alpha = alpha
        markersize = markersize
        lw = lw
        color = color[0]
        markeredgecolor = color

        nums = [0, 17, 22, 27, 31, 36, 42, 48, 60, 68]

        # close eyes and mouths
        plot_close = lambda i1, i2: plt.plot([pts[i][0, i1], pts[i][0, i2]], [pts[i][1, i1], pts[i][1, i2]],
                                                color=color, lw=lw, alpha=alpha - 0.1)
        plot_close(41, 36)
        plot_close(47, 42)
        plot_close(59, 48)
        plot_close(67, 60)

        for ind in range(len(nums) - 1):
            l, r = nums[ind], nums[ind + 1]
            plt.plot(pts[i][0, l:r], pts[i][1, l:r], color=color, lw=lw, alpha=alpha - 0.1)

            plt.plot(pts[i][0, l:r], pts[i][1, l:r], marker='o', linestyle='None', markersize=markersize,
                        color=color,
                        markeredgecolor=markeredgecolor, alpha=alpha)


def draw_axis(img, yaw, pitch, roll, tdx=None, tdy=None, size = 100, pts68=None):
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    if tdx != None and tdy != None:
        tdx = tdx
        tdy = tdy
    else:
        height, width = img.shape[:2]
        tdx = width / 2
        tdy = height / 2

    tdx = pts68[0,30]
    tdy = pts68[1,30]


    minx, maxx = np.min(pts68[0, :]), np.max(pts68[0, :])
    miny, maxy = np.min(pts68[1, :]), np.max(pts68[1, :])
    llength = sqrt((maxx - minx) * (maxy - miny))
    size = llength * 0.5


    # if pts8 != None:
    #     tdx = 

    # X-Axis pointing to right. drawn in red
    x1 = size * (cos(yaw) * cos(roll)) + tdx
    y1 = size * (cos(pitch) * sin(roll) + cos(roll) * sin(pitch) * sin(yaw)) + tdy

    # Y-Axis | drawn in green
    #        v
    x2 = size * (-cos(yaw) * sin(roll)) + tdx
    y2 = size * (cos(pitch) * cos(roll) - sin(pitch) * sin(yaw) * sin(roll)) + tdy

    # Z-Axis (out of the screen) drawn in blue
    x3 = size * (sin(yaw)) + tdx
    y3 = size * (-cos(yaw) * sin(pitch)) + tdy

    minus=0

    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x1),int(y1)),(0,0,255),4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x2),int(y2)),(0,255,0),4)
    cv2.line(img, (int(tdx), int(tdy)-minus), (int(x3),int(y3)),(255,0,0),4)

    return img

def crop_img(img, roi_box):
    h, w = img.shape[:2]

    sx, sy, ex, ey = [int(round(_)) for _ in roi_box]
    dh, dw = ey - sy, ex - sx
    if len(img.shape) == 3:
        res = np.zeros((dh, dw, 3), dtype=np.uint8)
    else:
        res = np.zeros((dh, dw), dtype=np.uint8)
    if sx < 0:
        sx, dsx = 0, -sx
    else:
        dsx = 0

    if ex > w:
        ex, dex = w, dw - (ex - w)
    else:
        dex = dw

    if sy < 0:
        sy, dsy = 0, -sy
    else:
        dsy = 0

    if ey > h:
        ey, dey = h, dh - (ey - h)
    else:
        dey = dh

    res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
    return res

def write_obj(obj_name, vertices, triangles):
    triangles = triangles.copy() # meshlab start with 1

    if obj_name.split('.')[-1] != 'obj':
        obj_name = obj_name + '.obj'

    # write obj
    with open(obj_name, 'w') as f:
        # write vertices & colors
        for i in range(vertices.shape[1]):
            s = 'v {:.4f} {:.4f} {:.4f}\n'.format(vertices[0, i], vertices[1, i], vertices[2, i])
            f.write(s)
        # write f: ver ind/ uv ind
        for i in range(triangles.shape[1]):
            s = 'f {} {} {}\n'.format(triangles[2, i], triangles[1, i], triangles[0, i])
            f.write(s)