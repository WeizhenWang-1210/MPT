import torch
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ref: https://waymo.com/open/challenges/2021/motion-prediction/
class Metric():
    
    def __init__(self, pred, truth, probas):
        # shape: (batch_size, n_modes)
        self.probas = probas
        
        # shape: (batch_size, sec * freq, n-dim)
        self.truth = truth
        
        # euclean dist diff at the time prediction starts
        pos_diff = torch.sqrt(torch.sum((truth[:, -1, :] - truth[:, -2, :]) ** 2, dim=1))
        self.init_v = pos_diff * 10
        self.alpha = (self.init_v - 1.4) / (11 - 1.4)
        self.missrate_scale = torch.ones_like(self.init_v)
        for i, (v, a) in enumerate(zip(self.init_v, self.alpha)):
            if v < 1.4:
                self.missrate_scale[i] = 0.5
            elif v < 11:
                self.missrate_scale[i] = 0.5 + 0.5 * a
        
        # shape: (batch_size, n_modes, sec * freq, n-dim)
        self.pred = pred
        
        # shape: (batch_size)
        self.top_nxt_idx = torch.argmax(probas, dim=1)
        
        self.batch_size = self.probas.shape[0]
        self.n_modes = self.probas.shape[1]
        self.n_agents = self.pred.shape[3]
        
        self.miss_rate_threshold_table = {
            29: {'x': 2, 'y': 1},
            49: {'x': 3.6, 'y': 1.8},
            79: {'x': 6, 'y': 3}
        }
    # minimum displacement error of all predicted trajectories
    # averaged across all time steps, (in our case) across all batches
    def minADE(self):
        # diff from ground truth of each traj pred
        diffs = torch.stack([self.pred[:, i, :, :] - self.truth for i in range(self.n_modes)], dim=1)
        diff_norm = torch.norm(diffs, p=2, dim=3)
        diff_norm = torch.mean(diff_norm, dim=(0, 2))
        return torch.min(diff_norm)
    
    # minimum final displacement error
    def minFDE(self):
        # diff from ground truth of each traj pred
        diffs = torch.stack([self.pred[:, i, -1:, :] - self.truth[:, -1:, :] for i in range(self.n_modes)], dim=1)
        diff_norm = torch.norm(diffs, p=2, dim=3)
        diff_norm = torch.mean(diff_norm, dim=(0, 2))
        return torch.min(diff_norm)
    
    # u @ R = truth   |   u is unit vector, R is rotation matrix
    # A @ X = b  --->   min_X ||AX - b||
    # miss rate
    def missRate(self, T=29):
        truth = self.truth[:, T, :]
        # denominator = torch.norm(truth, p=2, dim=1, keepdim=True)
        # u = truth / denominator
        u = torch.nn.functional.normalize(truth, p=2.0)
        
        # iterate thru batch, later averaged over batch
        not_missed = 0
        for b in range(self.batch_size):
            R = torch.linalg.lstsq(
                u[b, :].unsqueeze(0), 
                truth[b, :].unsqueeze(0)).solution
            diff = truth[b, :] - self.pred[b, :, T, :]
            pred = diff @ R
            x_in_threshold = pred[:, 0] < self.missrate_scale[b] * self.miss_rate_threshold_table[T]['x']
            y_in_threshold = pred[:, 1] < self.missrate_scale[b] * self.miss_rate_threshold_table[T]['y']
            both_in = torch.logical_and(x_in_threshold, y_in_threshold)
            if torch.any(both_in):
                not_missed += 1
        return 1 - (not_missed / self.batch_size)
    
    def allMissRates(self):
        times = self.miss_rate_threshold_table.keys()
        missRates = np.array([self.missRate(t) for t in times])
        return np.mean(missRates)

def get_corners(x, y, h, w, ori):
    corners = np.array([
        [-w/2, h/2],
        [ w/2, h/2],
        [-w/2, -h/2],
        [ w/2, -h/2]
    ])
    if ori is not None:
        ori = ori.detach().cpu().numpy()
        c, s = np.cos(ori), np.sin(ori)
        R = np.array(((c, -s), (s, c))).reshape(2, 2)
        corners = corners @ R.T + np.array([x, y])
    else:
        corners += np.array([x, y])
    return corners
    
class Visualizer():
    
    def __init__(self, input, probas, coordinates):
        # input to model dumped here, it is a dictionary
        self.input = input
        
        # output of model
        self.probas = probas # shape: (batch_size, n_modes)
        self.coordinates = coordinates # shape: (batch_size, n_modes, sec * freq, n-dim)
        
        # shape: (batch_size)
        self.top_nxt_idx = torch.argmax(probas, dim=1)
        
        # shape: (batch_size, sec * freq, n-dim)
        self.top_pred = []
        for b, best_idx in zip(self.coordinates, self.top_nxt_idx):
            self.top_pred.append(b[best_idx, :, :])
        self.top_pred = torch.stack(self.top_pred)
        
        self.batch_size = self.probas.shape[0]
        self.n_modes = self.probas.shape[1]
        self.n_agents = self.coordinates.shape[3]
        
        self.true_color = 'green'
        self.pred_color = 'blue'
    
    def draw_seg_corners(self, c0, c1):
        x = np.array([c0[0], c1[0]])
        y = np.array([c0[1], c1[1]])
        plt.plot(x, y, color="black", linewidth=0.6, alpha=0.8)
    
    def draw_obj_box(self, corners):
        c0, c1, c2, c3 = corners[0], corners[1], corners[2], corners[3]
        self.draw_seg_corners(c0, c1) # upper line
        self.draw_seg_corners(c1, c3) # right line
        self.draw_seg_corners(c3, c2) # bottom line
        self.draw_seg_corners(c2, c0) # left line

    def visualize_one_traj_target(self, batch_id=0, traj_id=0):
        assert batch_id >= 0, batch_id < self.batch_size
        assert traj_id >= 0, traj_id < self.n_modes
        hist_pos = self.input['target/history/xy'][batch_id, :, :].detach().cpu().numpy()
        fut_pred_pos = self.coordinates[batch_id, traj_id, :, :].detach().cpu().numpy()
        fut_true_pos = self.input['target/future/xy'][batch_id, :, :].detach().cpu().numpy()
        true_traj = np.concatenate((hist_pos, fut_true_pos), axis=0)
        pred_traj = np.concatenate((hist_pos, fut_pred_pos), axis=0)
        target_w = self.input['target/width'][batch_id]
        target_h = self.input['target/length'][batch_id]
        ori = self.input['target/future/yaw'][batch_id, 0, :]
        self.draw_obj_box(
            corners=get_corners(hist_pos[-1, 0], hist_pos[-1, 1], target_w, target_h, ori))
        plt.plot(true_traj[:, 0], true_traj[:, 1], color=self.true_color, zorder=1, label='Main Traj - True')
        plt.plot(pred_traj[:, 0], pred_traj[:, 1], color=self.pred_color, zorder=1, label='Main Traj - Pred')
        plt.scatter(hist_pos[-1, 0], hist_pos[-1, 1], s=1.5, color="yellow", marker="*", zorder=2, label="Main Agent")
    
    def visualize_targets(self):
        for b in range(self.batch_size):
            valid = (torch.all(self.input["target/future/valid"][b]) \
                and torch.all(self.input["target/history/valid"][b]))
            if valid:
                self.visualize_one_traj_target(b, self.top_nxt_idx[b])
    
    def road_features(self):
        segments=self.input["road_network_segments"]
        # to see the points, uncomment the line below
        # plt.scatter(segments[:, 0, 0], segments[:, 0, 1], color="black", s=0.1, zorder=0)
        segs = []
        new_seg = True
        for seg in segments[:, 0, :]:
            if new_seg:
                segs.append(seg)
                new_seg = False
            elif torch.sum((segs[-1] - seg) ** 2) < 15:
                segs.append(seg)
            else:
                segs = torch.stack(segs)
                plt.plot(segs[:, 0], segs[:, 1], linewidth=0.3, color="black", alpha=0.6, zorder=0)
                segs = []
                new_seg = True
    
    def one_other(self, id):
        
        # (num_others, seq * freq, dim)
        hist_pos = self.input['other/history/xy'][id]
        fut_pos = self.input['other/future/xy'][id]
        traj = np.concatenate((hist_pos, fut_pos), axis=0)
        
        # we only need ori at step 0, since we only plot the bounding box at first step
        ori = self.input['other/future/yaw'][id][0]        
        w = self.input['other/width'][id]
        h = self.input['other/length'][id]
        self.draw_obj_box(
            corners=get_corners(hist_pos[-1, 0], hist_pos[-1, 1], w, h, ori))
        plt.plot(traj[:, 0], traj[:, 1], color='pink', zorder=1, alpha=0.75, label="other trajs - true")
        plt.scatter(hist_pos[-1, 0], hist_pos[-1, 1], s=0.3, color="orange", marker=".", zorder=2, label="Others")
        
    def all_others(self):
        for i in range(self.input['other/width'].shape[0]):
            valid = (torch.all(self.input["other/future/valid"][i]) \
                and torch.all(self.input["other/history/valid"][i]))
            if valid:
                self.one_other(i)
    
    def save(self, file_name='test_vis.jpg'):
        legend_elements = [
            Line2D([0], [0], color='pink', lw=1, alpha=0.75, label='other trajs - true'),
            Line2D([0], [0], color=self.true_color, lw=1, alpha=1, label='Main Traj - True'),
            Line2D([0], [0], color=self.pred_color, lw=1, alpha=1, label='Main Traj - Pred'),
            Line2D([0], [0], marker='*', color='white', markerfacecolor='yellow', label='Main Agent', markersize=10),
            Line2D([0], [0], marker='.', color='white', markerfacecolor='orange', label='Others', markersize=10),
        ]
        plt.legend(handles=legend_elements, loc='lower right')
        plt.savefig(file_name, dpi=600)
    
    def reset_fig(self):
        plt.figure()