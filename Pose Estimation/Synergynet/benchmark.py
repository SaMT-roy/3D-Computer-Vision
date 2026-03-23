from model import *
from utils import *
import torch.utils.data as data
import config as cfg
import time
from utils import _load
import os.path as osp
import glob
import os
from math import cos, atan2, asin,sqrt

device='mps'
param_pack = ParamsPack()
img_list = sorted(glob.glob('aflw2000_data/AFLW2000-3D_crop/*.jpg'))

d = 'aflw2000_data/eval'
yaws_list = _load(osp.join(d, 'AFLW2000-3D.pose.npy'))
pts68_all_ori = _load(osp.join(d, 'AFLW2000-3D.pts68.npy')) # origin
pts68_all_re = _load(osp.join(d, 'AFLW2000-3D-Reannotated.pts68.npy')) # reannonated
roi_boxs = _load(osp.join(d, 'AFLW2000-3D_crop.roi_box.npy'))

class DDFATestDataset(data.Dataset):
    def __init__(self, filelists, root='', transform=None):
        self.root = root
        self.transform = transform
        self.lines = Path(filelists).read_text().strip().split('\n')

    def __getitem__(self, index):
        path = osp.join(self.root, self.lines[index])
        img = img_loader(path)

        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.lines)
    
def extract_param(root='', 
                  filelists=None, 
                  device='mps', 
                  batch_size=cfg.batch_size,
                  saved_path=cfg.trained_model_path):

    model = SynergyNet()
    model.load_state_dict(torch.load(saved_path))
    model.to(device)
    model.eval()

    dataset = DDFATestDataset(filelists=filelists, root=root, transform=Compose_GT([ToTensor()]))
    data_loader = data.DataLoader(dataset, batch_size=batch_size)

    end = time.time()
    outputs = []
    with torch.no_grad():
        for _, inputs in enumerate(data_loader):
            inputs = inputs.to(device)
            output = model.forward_test(inputs)

            for i in range(output.shape[0]):
              param_prediction = output[i].cpu().numpy().flatten()
              outputs.append(param_prediction)
        outputs = np.array(outputs, dtype=np.float32)

    print('Extracting params take {: .3f}s'.format(time.time() - end))
    return outputs, model.data_param

def ana_alfw2000(nme_list):

    leng = nme_list.shape[0]
    yaw_list_abs = np.abs(yaws_list)[:leng]
    ind_yaw_1 = yaw_list_abs <= 30
    ind_yaw_2 = np.bitwise_and(yaw_list_abs > 30, yaw_list_abs <= 60)
    ind_yaw_3 = yaw_list_abs > 60

    nme_1 = nme_list[ind_yaw_1]
    nme_2 = nme_list[ind_yaw_2]
    nme_3 = nme_list[ind_yaw_3]

    mean_nme_1 = np.mean(nme_1) * 100
    mean_nme_2 = np.mean(nme_2) * 100
    mean_nme_3 = np.mean(nme_3) * 100

    std_nme_1 = np.std(nme_1) * 100
    std_nme_2 = np.std(nme_2) * 100
    std_nme_3 = np.std(nme_3) * 100

    mean_all = [mean_nme_1, mean_nme_2, mean_nme_3]
    mean = np.mean(mean_all)
    std = np.std(mean_all)

    s0 = '\nFacial Alignment on AFLW2000-3D (NME):'
    s1 = '[ 0, 30]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_1, std_nme_1)
    s2 = '[30, 60]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_2, std_nme_2)
    s3 = '[60, 90]\tMean: {:.3f}, Std: {:.3f}'.format(mean_nme_3, std_nme_3)
    s4 = '[ 0, 90]\tMean: {:.3f}, Std: {:.3f}'.format(mean, std)

    s = '\n'.join([s0, s1, s2, s3, s4])

    return  s

def calc_nme_alfw2000(pts68_fit_all, option='ori'):
    if option == 'ori':
        pts68_all = pts68_all_ori
    elif option == 're':
        pts68_all = pts68_all_re
    std_size = 120

    nme_list = []
    length_list = []
    for i in range(len(roi_boxs)):
        pts68_fit = pts68_fit_all[i]
        pts68_gt = pts68_all[i]

        sx, sy, ex, ey = roi_boxs[i]
        scale_x = (ex - sx) / std_size
        scale_y = (ey - sy) / std_size
        pts68_fit[0, :] = pts68_fit[0, :] * scale_x + sx
        pts68_fit[1, :] = pts68_fit[1, :] * scale_y + sy

        # build bbox
        minx, maxx = np.min(pts68_gt[0, :]), np.max(pts68_gt[0, :])
        miny, maxy = np.min(pts68_gt[1, :]), np.max(pts68_gt[1, :])
        llength = sqrt((maxx - minx) * (maxy - miny))
        length_list.append(llength)

        dis = pts68_fit - pts68_gt[:2, :]
        dis = np.sqrt(np.sum(np.power(dis, 2), 0))
        dis = np.mean(dis)
        nme = dis / llength
        nme_list.append(nme)

    nme_list = np.array(nme_list, dtype=np.float32)
    return nme_list


def _benchmark_aflw2000(outputs):
    '''Calculate the error statistics.'''
    return ana_alfw2000(calc_nme_alfw2000(outputs,option='ori'))

def matrix2angle(R):
    '''convert matrix to angle'''
    if R[2, 0] != 1 and R[2, 0] != -1:
        x = asin(R[2, 0])
        y = atan2(R[1, 2] / cos(x), R[2, 2] / cos(x))
        z = atan2(R[0, 1] / cos(x), R[0, 0] / cos(x))

    else:  # Gimbal lock
        z = 0  # can be anything
        if R[2, 0] == -1:
            x = np.pi / 2
            y = z + atan2(R[0, 1], R[0, 2])
        else:
            x = -np.pi / 2
            y = -z + atan2(-R[0, 1], -R[0, 2])
    
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi

    return [rx, ry, rz]

def P2sRt(P):
    '''decomposing camera matrix P'''   
    t3d = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2)) / 2.0
    r1 = R1 / np.linalg.norm(R1)
    r2 = R2 / np.linalg.norm(R2)
    r3 = np.cross(r1, r2)
    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t3d

def parse_pose(param):
    '''parse parameters into pose'''
    if len(param)==62:
        param = param * param_pack.param_std[:62] + param_pack.param_mean[:62]
    else:
        param = param * param_pack.param_std + param_pack.param_mean
    Ps = param[:12].reshape(3, -1)  # camera matrix
    s, R, t3d = P2sRt(Ps)
    P = np.concatenate((R, t3d.reshape(3, -1)), axis=1)  # without scale
    pose = matrix2angle(R)  # yaw, pitch, roll
    return P, pose

def parsing(param):
    p_ = param[:, :12].reshape(-1, 3, 4)
    p = p_[:, :, :3]
    offset = p_[:, :, -1].reshape(-1, 3, 1)
    alpha_shp = param[:, 12:52].reshape(-1, 40, 1)
    alpha_exp = param[:, 52:62].reshape(-1, 10, 1)
    return p, offset, alpha_shp, alpha_exp

def reconstruct_vertex(param, data_param, whitening=True, transform=True, lmk_pts=68):
    """
    This function includes parameter de-whitening, reconstruction of landmarks, and transform from coordinate space (x,y) to image space (u,v)
    """
    param_mean, param_std, w_shp_base, u_base, w_exp_base = data_param

    if whitening:
        if param.shape[1] == 62:
            param = param * param_std[:62] + param_mean[:62]
        else:
            raise NotImplementedError("Parameter length must be 62")

    if param.shape[1] == 62:
        p, offset, alpha_shp, alpha_exp = parsing(param)
    else:
        raise NotImplementedError("Parameter length must be 62")

    vertex = p @ (u_base + w_shp_base @ alpha_shp + w_exp_base @ alpha_exp).contiguous().view(-1, lmk_pts, 3).transpose(1,2) + offset
    if transform:
        vertex[:, 1, :] = param_pack.std_size + 1 - vertex[:, 1, :]

    return vertex

def benchmark_aflw2000_params(params, data_param):
    '''Reconstruct the landmark points and calculate the statistics'''
    outputs = []
    params = torch.Tensor(params)

    batch_size = 50
    num_samples = params.shape[0]
    iter_num = math.floor(num_samples / batch_size)
    residual = num_samples % batch_size
    for i in range(iter_num+1):
        if i == iter_num:
            if residual == 0:
                break
            batch_data = params[i*batch_size: i*batch_size + residual]
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.cpu().numpy()
            for j in range(residual):
                outputs.append(lm[j, :2, :])
        else:
            batch_data = params[i*batch_size: (i+1)*batch_size]
            lm = reconstruct_vertex(batch_data, data_param, lmk_pts=68)
            lm = lm.cpu().numpy()
            for j in range(batch_size):
                if i == 0:
                    #plot the first 50 samples for validation
                    bkg = cv2.imread(img_list[i*batch_size+j],-1)
                    lm_sample = lm[j]
                    c0 = np.clip((lm_sample[1,:]).astype(np.int64), 0, 119)
                    c1 = np.clip((lm_sample[0,:]).astype(np.int64), 0, 119)
                    for y, x, in zip([c0,c0,c0-1,c0-1],[c1,c1-1,c1,c1-1]):
                        bkg[y, x, :] = np.array([233,193,133])
                    cv2.imwrite(f'./results/{i*batch_size+j}.png', bkg)
                outputs.append(lm[j, :2, :])
    return _benchmark_aflw2000(outputs)


def benchmark_FOE(params):
    """
    FOE benchmark validation. Only calculate the groundtruth of angles within [-99, 99] (following FSA-Net https://github.com/shamangary/FSA-Net)
    """

    # AFLW200 groundturh and indices for skipping, whose yaw angle lies outside [-99, 99]
    exclude_aflw2000 = 'aflw2000_data/eval/ALFW2000-3D_pose_3ANG_excl.npy'
    skip_aflw2000 = 'aflw2000_data/eval/ALFW2000-3D_pose_3ANG_skip.npy'

    if not os.path.isfile(exclude_aflw2000) or not os.path.isfile(skip_aflw2000):
        raise RuntimeError('Missing data')

    pose_GT = np.load(exclude_aflw2000) 
    skip_indices = np.load(skip_aflw2000)
    pose_mat = np.ones((pose_GT.shape[0],3))

    idx = 0
    for i in range(params.shape[0]):
        if i in skip_indices:
            continue
        P, angles = parse_pose(params[i])
        angles[0], angles[1], angles[2] = angles[1], angles[0], angles[2] # we decode raw-ptich-yaw order
        pose_mat[idx,:] = np.array(angles)
        idx += 1

    pose_analyis = np.mean(np.abs(pose_mat-pose_GT),axis=0) # pose GT uses [pitch-yaw-roll] order
    MAE = np.mean(pose_analyis)
    yaw = pose_analyis[1]
    pitch = pose_analyis[0]
    roll = pose_analyis[2]
    msg = 'Mean MAE = %3.3f (in deg), [yaw,pitch,roll] = [%3.3f, %3.3f, %3.3f]'%(MAE, yaw, pitch, roll)
    print('\nFace orientation estimation:')
    print(msg)
    return msg

def benchmark():
    '''benchmark validation pipeline'''

    def aflw2000():
        root = 'aflw2000_data/AFLW2000-3D_crop'
        filelists = 'aflw2000_data/AFLW2000-3D_crop.list'

        if not os.path.isdir(root) or not os.path.isfile(filelists):
            raise RuntimeError('check if the testing data exist')

        params, data_param = extract_param(
            root=root,
            filelists=filelists)

        info_out_fal = benchmark_aflw2000_params(params, data_param)
        print(info_out_fal)
        info_out_foe = benchmark_FOE(params)

    aflw2000()