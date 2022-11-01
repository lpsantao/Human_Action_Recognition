import sys
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import pickle
from numpy.lib.format import open_memmap
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)
np.set_printoptions(threshold=sys.maxsize)



# for benchmark Cross-Subject (xsub)
training_subjects = [1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38]

class Config():
    def __init__(self):
        self.max_frame = 300   # the maximum length of frames
        self.joint_n = 25   # the number of joints
        self.joint_d = 3    # the dimension of joints
        self.max_skeletons = 2      # the maximum of skeletons in 1 sample
        self.missing_data_dir = 'D:\\datasets\\NTU_RGBD_samples_with_missing_skeletons.txt'
        self.skeletons_dir = 'D:\\datasets\\nturgb+d_skeletons\\'
        self.skeletons_dir_debug = 'D:\\datasets\\debug_ntu\\'
        self.save_dir = 'D:\\datasets\\saved_NTU_RGB_D\\'


def read_skeleton(ske_name_dir, joint_n, max_skel):
    '''
    Reads .skeleton file
    :param ske_name_dir: path to each file .skeleton
                         ex: 'C:\\Users\\Liliana\\Downloads\\datasets\\debug_ntu\\S001C001P001R001A001.skeleton'
    :param joint_n: maxumum number of joints per skeleton as per Config()
    :param max_skel: maximum number of skeletons in file as per Config()
    :return: dataframe with .skeleton and 4D array [3 (x,y,z),max_frames,max_joints,max_skeletons]
    '''
    skeleton_file = dict()
    joint_list_plot = dict()
    with open(ske_name_dir, 'r') as file:
        skeleton_file = dict()
        skeleton_file['frame_count'] = int(file.readline())
        skeleton_file['frame_info'] = []
        for frame in range(skeleton_file['frame_count']):
            frame_info = dict()
            frame_info['body_count'] = int(file.readline())
            frame_info['body_info'] = []
            for body in range(frame_info['body_count']):
                body_info = dict()
                body_info_params = ['body_ID', 'cliped_edges', 'left_hand_confidence', 'left_hand_state',
                                    'right_hand_confidence', 'right_hand_state', 'is_restricted', 'lean_x',
                                    'lean_y', 'is_tracked']
                body_info = {info: float(par) for info, par in zip(body_info_params, file.readline().split())}
                body_info['joints_count'] = int(file.readline())
                body_info['joints_info'] = []
                for v in range(body_info['joints_count']):
                    joint_info_key = ['x', 'y', 'z', 'depth_x', 'depth_y', 'color_x', 'color_y', 'rw', 'rx', 'ry',
                                      'rz', 'tracking_state']
                    joint_info = {k: float(v) for k, v in zip(joint_info_key, file.readline().split())}
                    body_info['joints_info'].append(joint_info)
                    if frame == 0:
                        joint_list_plot[v] = [joint_info['x'], joint_info['y'], joint_info['z']]
                frame_info['body_info'].append(body_info)
            skeleton_file['frame_info'].append(frame_info)

        skeleton_file_df = pd.DataFrame(skeleton_file)

        data = np.zeros((3, skeleton_file['frame_count'], joint_n, max_skel))
        for n, f in enumerate(skeleton_file['frame_info']):
            for m, b in enumerate(f['body_info']):
                for j, v in enumerate(b['joints_info']):
                    if m < max_skel and j < joint_n:
                        data[:, n, j, m] = [v['x'], v['y'], v['z']]
                    else:
                        pass

    return skeleton_file_df, data, joint_list_plot


def generate_splits(skel_dir, mdata_dir, out_path, max_f, max_body, num_joint, split='training'):

    missing_list = []
    sample_name = []
    sample_label = []

    skeletons_missing = np.genfromtxt(mdata_dir, dtype='str')
    for j in skeletons_missing:
        missing_list.append(j + '.skeleton')

    for skeleton_file in os.listdir(skel_dir):
        if skeleton_file in missing_list:
            continue
        action_label = int(skeleton_file[skeleton_file.find('A') + 1:skeleton_file.find('A') + 4])
        id_subject = int(skeleton_file[skeleton_file.find('P') + 1:skeleton_file.find('P') + 4])
        # replic_number = int(skeleton_file[skeleton_file.find('R') + 1:skeleton_file.find('R') + 4])
        # id_camera = int(skeleton_file[skeleton_file.find('C') + 1:skeleton_file.find('C') + 4])

        training = (id_subject in training_subjects)

        if split == 'training':
            sample = training
        elif split == 'validation':
            sample = not training
        else:
            raise ValueError()

        if sample:
            sample_name.append(skeleton_file)
            sample_label.append(action_label - 1)

    pickle.dump((sample_name, list(sample_label)), open('{}/{}_label.pkl'.format(out_path, split), 'wb'))

    fp = open_memmap('{}/{}_data.npy'.format(out_path, split), dtype='float32', mode='w+',
                     shape=(len(sample_label), 3, max_f, num_joint, max_body))
    print((len(sample_label), 3, max_f, num_joint, max_body))

    for i, s in enumerate(tqdm(sample_name)):
        df, data, list_pl = read_skeleton(os.path.join(skel_dir, s), max_skel=max_body, joint_n=num_joint)
        fp[i, :, 0:data.shape[1], :, :] = data


def draw_skeleton(joints_list, outdir, outname, show=0, save=1, autoscale=1):
    '''
    Draws skeleton joints and connections
    :param joints_list: joint list per frame as: {0:[x,y,z], 1:[x,y,z]... joint_number:[x,y,z]}
    :param outdir: output path
    :param outname: output name for image
    :param show: 1 if to show as plot
    :param save: 1 if to save as png file
    :param autoscale: 1 if autoscale
    '''
    lcolor = "#3498db"
    rcolor = "#e74c3c"
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if autoscale:
        xx = []
        yy = []
        zz = []
    for jj in range(0, len(joints_list)):
        for num, joint in joints_list.items():
            ax.scatter(joint[0], joint[1], joint[2], color=lcolor, s=5)
            if autoscale:
                xx.append(joint[0])
                yy.append(joint[1])
                zz.append(joint[2])

        connectivity = [(0, 1), (1, 20), (20, 2), (2, 3), (20, 8), (8, 9), (9, 10), (10, 11), (11, 24), (24, 23),
                        (20, 4), (4, 5), (5, 6), (6, 7), (7, 22), (22, 21), (0, 16), (16, 17), (17, 18), (18, 19),
                        (0, 12), (12, 13), (13, 14), (14, 15)]

        for connection in connectivity:
            t = connection[0]
            f = connection[1]

            ax.plot([joints_list[f][0], joints_list[t][0]], [joints_list[f][1], joints_list[t][1]], [joints_list[f][2], joints_list[t][2]], lw=2, color=rcolor)

        if autoscale:
            ax.set_xlabel('X Label')
            ax.set_xlim(np.min(xx)-0.1, np.max(xx)+0.1)
            ax.set_ylabel('Y Label')
            ax.set_ylim(np.min(yy)-0.1, np.max(yy)+0.1)
            ax.set_zlabel('Z Label')
            ax.set_zlim(np.min(zz)-0.1, np.max(zz)+0.1)
        else:
            ax.set_xlabel('X Label')
            ax.set_xlim(-0.6, 0.6)
            ax.set_ylabel('Y Label')
            ax.set_ylim(-1, 1)
            ax.set_zlabel('Z Label')
            ax.set_zlim(-3, 5)

        ax.view_init(elev=-90., azim=90)

    if show:
        plt.show()

    if save:
        plt.savefig(outdir + outname + ".png")


if __name__ == '__main__':
    split = ['training', 'validation']
    C = Config()
    for s in split:
        generate_splits(C.skeletons_dir, C.missing_data_dir, C.save_dir, C.max_frame, C.max_skeletons,
                        C.joint_n, split=s)
    '''
        filename = 'S001C001P001R001A001.skeleton'
    a, b, plot = read_skeleton(os.path.join(C.skeletons_dir, filename), C.joint_n, C.max_skeletons)
    draw_skeleton(plot, C.save_dir, 'skel_plot')

    '''


