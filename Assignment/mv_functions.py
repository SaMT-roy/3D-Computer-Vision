import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def normalize_points(points, target_dist):
    """
    Hartley's Normalization
    """
    dim = points.shape[1]
    centroid = np.mean(points, axis=0)
    translated_points = points - centroid
    
    # Calculate average distance from origin
    avg_dist = np.mean(np.linalg.norm(translated_points, axis=1))
    
    # Prevent division by zero if all points are identical
    scale = target_dist / avg_dist if avg_dist > 0 else 1.0 
    
    normalized_points = translated_points * scale
    
    # Build the transformation matrix T (Homogeneous coordinates)
    T = np.eye(dim + 1)
    for i in range(dim):
        T[i, i] = scale
        T[i, dim] = -scale * centroid[i]
        
    return normalized_points, T

def reprojection_error(params, pts_3d, pts_2d, K, confidences):
    """
    Cost function for Levenberg-Marquardt optimization.
    Calculates the exact pixel distance between the mathematical projection 
    and the noisy observed pixels.
    """
    # Unpack parameters: First 3 are rotation vector (Rodrigues), last 3 are translation
    rvec = params[:3]
    tvec = params[3:]
    
    # Convert rotation vector to 3x3 matrix
    R = Rotation.from_rotvec(rvec).as_matrix()
    
    errors = []
    for i in range(len(pts_3d)):
        # 1. Rigid Body Transform: X_cam = R * X_3d + t
        X_cam = R @ pts_3d[i] + tvec
        
        # 2. Camera Projection: x_img = K * X_cam
        x_img = K @ X_cam
        
        # 3. Dehomogenize (Divide by Z to get pixel coordinates u, v)
        u_proj = x_img[0] / x_img[2]
        v_proj = x_img[1] / x_img[2]
        
        # 4. Calculate error, weighted by the point's confidence
        err_u = (u_proj - pts_2d[i][0]) * confidences[i]
        err_v = (v_proj - pts_2d[i][1]) * confidences[i]
        
        errors.extend([err_u, err_v])
        
    return np.array(errors)

def estimate_pose(camera_observations, all_3d_points, K_matrix, noise_factor=0):

    K = np.array(K_matrix)
    K_inv = np.linalg.inv(K)
    
    # Occlusion Handling Extract only visible points and drop nulls. No interpolation used.
    valid_3d = []
    valid_2d_pixels = []
    confidences = []
    
    for obs in camera_observations:
        if obs["visible"] and obs["u"] is not None and obs["v"] is not None:
            valid_3d.append(all_3d_points[obs["point_index"]])
            valid_2d_pixels.append([obs["u"], obs["v"]])
            confidences.append(obs["confidence"])
            
    pts_3d = np.array(valid_3d)
    pts_2d_pixels = np.array(valid_2d_pixels) + np.random.normal(0, noise_factor)
    
    # We solve for X_3d_cam = R * X_3d + t
    # [u,v] --> [u,v,1]
    pts_2d_homo = np.hstack((pts_2d_pixels, np.ones((len(pts_2d_pixels), 1))))
    
    # pixel_2d = K * camera_3d
    pts_2d_cam = (K_inv @ pts_2d_homo.T).T[:, :2] # Keep only x,y. Z is implicitly 1.
    
    # Hartley Normalization :Centers and scales data to prevent numerical instability during SVD.
    pts_2d_norm, T_2d = normalize_points(pts_2d_cam, np.sqrt(2))
    pts_3d_norm, T_3d = normalize_points(pts_3d, np.sqrt(3))
    
    # Step 3: Build Matrix A for Direct Linear Transform (DLT)
    N = len(pts_3d_norm)
    A = np.zeros((2 * N, 12))
    
    for i in range(N):
        X, Y, Z = pts_3d_norm[i]
        u, v = pts_2d_norm[i]
        
        # Row 1 for x-axis equation
        A[2*i, 0:4] = [-X, -Y, -Z, -1]
        A[2*i, 8:12] = [u*X, u*Y, u*Z, u]
        
        # Row 2 for y-axis equation
        A[2*i+1, 4:8] = [-X, -Y, -Z, -1]
        A[2*i+1, 8:12] = [v*X, v*Y, v*Z, v]

    # Solve A*p = 0 using Singular Value Decomposition
    _, _, Vt = np.linalg.svd(A)
    p_norm = Vt[-1] # The solution is the last row of V^T (smallest singular value)
    P_norm = p_norm.reshape(3, 4)
    
    # Denormalize P : Convert P back to our original (un-normalized) coordinate space
    # We did : pts_2d_norm = T_2d_norm * pts_2d, pts_3d_norm = T_3d_norm * pts_3d

    # In DLT we did : 
    # 1. pts_2d_norm = Pose * pts_3d_norm 
    # 2. T_2d_norm * pts_2d = Pose * T_3d_norm * pts_3d
    # 3. pts_2d = (T_2d_norm)^-1 * Pose * T_3d_norm * pts_3d
    # 4. pts_2d = P_raw * pts_3d
    P_raw = np.linalg.inv(T_2d) @ P_norm @ T_3d
    
    # Extract raw Rotation and Translation from P
    R_raw = P_raw[:, :3]
    t_raw = P_raw[:, 3]

    # Reflection Check: If determinant is -1
    if np.linalg.det(R_raw) < 0:
        R_raw = -R_raw
        t_raw = -t_raw

    # Orthogonal Procrustes (Fixing the Rotation Matrix)
    # DLT doesn't guarantee R is orthogonal. We force it using SVD on R.
    U_r, S_r, Vt_r = np.linalg.svd(R_raw)
    R_est = U_r @ Vt_r

    # Recover translation scale (beta is the average of discarded singular values)
    beta = np.mean(S_r)
    t_est = t_raw / beta
        
    # --------------------------------------------------------------------------
    # Step 7: Non-Linear Refinement (Levenberg-Marquardt)
    # DLT minimized algebraic error. Now we minimize the true Geometric Error
    # measured in pixels, weighted by the observation confidence.
    # --------------------------------------------------------------------------
    # Convert R to a compact 3-element rotation vector for optimization
    initial_rvec = Rotation.from_matrix(R_est).as_rotvec()
    initial_params = np.hstack((initial_rvec, t_est))
    
    result = least_squares(
        reprojection_error, 
        initial_params, 
        args=(pts_3d, pts_2d_pixels, K, confidences),
        method='lm' # Levenberg-Marquardt
    )
    
    # --------------------------------------------------------------------------
    # Step 8: Formatting the Final Output
    # --------------------------------------------------------------------------
    final_rvec = result.x[:3]
    final_tvec = result.x[3:]
    final_R = Rotation.from_rotvec(final_rvec).as_matrix()
    
    return final_R, final_tvec



def triangulate_dlt(observations, P_matrices):
    """
    Constructs the linear system Ax = 0 from the projection rows.
    Solves for x via SVD. Explores the null-space of matrix A.
    """
    A = []
    for obs in observations:
        cam_idx, u, v, _ = obs
        P = P_matrices[cam_idx]
        
        # Cross product equations: x x PX = 0
        A.append(u * P[2, :] - P[0, :])
        A.append(v * P[2, :] - P[1, :])
        
    A = np.array(A)
    
    # Solve Ax = 0 via SVD
    U, S, Vt = np.linalg.svd(A)
    
    # The null-space solution is the last row of Vt
    X_homo = Vt[-1]
    
    # Dehomogenize (divide by W) to get 3D coordinates
    X_3d = X_homo[:3] / X_homo[3]
    return X_3d

def reprojection_cost(X_3d, observations, P_matrices):
    """
    Cost function for non-linear optimization.
    Calculates 2D pixel error weighted by confidence.
    """
    errors = []
    X_homo = np.append(X_3d, 1.0)
    
    for obs in observations:
        cam_idx, u, v, conf = obs
        P = P_matrices[cam_idx]
        
        # Project 3D point to 2D
        x_proj_homo = P @ X_homo
        u_proj = x_proj_homo[0] / x_proj_homo[2]
        v_proj = x_proj_homo[1] / x_proj_homo[2]
        
        # Apply Confidence Weighting!
        # Confidence acts as a scalar pulling the optimizer towards reliable data.
        errors.append((u_proj - u) * conf)
        errors.append((v_proj - v) * conf)
        
    return errors

def triangulate_optimal(initial_X, observations, P_matrices):
    """
    Minimizes the reprojection error using Levenberg-Marquardt (3 variables: X, Y, Z).
    """
    result = least_squares(
        reprojection_cost, 
        initial_X, 
        args=(observations, P_matrices),
        method='lm'
    )
    return result.x
