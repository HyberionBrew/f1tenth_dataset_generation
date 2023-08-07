# load in dataset, set terminals and compute the rewards
import sys
print(sys.executable)
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt # read pickle file from collect_dataset/dataset.pkl
from absl import app, flags
import csv
from scipy.spatial.distance import cdist
import pickle
import zarr
from tqdm import tqdm

import os

FLAGS = flags.FLAGS

flags.DEFINE_float('speed', 1.0, 'Mean speed of the car')
flags.DEFINE_string('input_folder', 'raw_datasets', 'Name of dataset file')
flags.DEFINE_string('output_file', 'SFTG_speed_1.0_processed.pkl', 'Name of dataset file')
flags.DEFINE_string('centerline_file', 'ellipse_map2.csv', 'Name of centerline file')
flags.DEFINE_bool('plot', True, 'Plot the dataset')
flags.DEFINE_integer('max_traj_length', 400, 'Maximum length of a trajectory')
flags.DEFINE_integer('min_traj_length', 3, 'Minimum length of a trajectory')


def lap_length(centerpoints):
    centerpoints = np.array(centerpoints)
    differences = np.diff(centerpoints, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return segment_lengths.sum()


class PickleDataset(object):
    def __init__(self, path, skip_loading=False):
        self.data = []
        # print the path
        print(f"Opening: {path}")
        with open(path, 'rb') as f:
            while True:
                try:
                    self.data.append(pkl.load(f))
                except EOFError:
                    break
        self.y_pose = [self.data[i][2]['poses_y'][0] for i in range(0, len(self.data))]
        self.x_pose = [self.data[i][2]['poses_x'][0] for i in range(0, len(self.data))]
        self.pose_points = np.stack((self.x_pose, self.y_pose), axis=-1)

        self.trajectories = []
        self.raw_trajectory = []
        if not(skip_loading):
            for i in range(len(self.data)):
                self.raw_trajectory.append(
                {'scan': np.array(self.data[i][2]['scans'][0]),
                'pose_x': self.data[i][2]['poses_x'][0],
                'pose_y': self.data[i][2]['poses_y'][0],
                'pose_theta': self.data[i][2]['poses_theta'][0],
                'linear_vel_y': self.data[i][2]['linear_vels_y'][0],
                'linear_vel_x': self.data[i][2]['linear_vels_x'][0],
                'ang_vels_z': self.data[i][2]['ang_vels_z'][0],
                'done': self.data[i][4],
                'agent': self.data[i][-1][0],
                'target_speed': self.data[i][-1][1],
                'steering': self.data[i][1],
                'velocity': self.data[i][0]})
                
    def get_pose(self):
        return self.x_pose, self.y_pose
    
    def get_done(self):
        return [datapoint['done'] for datapoint in self.raw_trajectory]
    

    # TODO! recompute all trajectories here
    def split_trajectories(self, progress):
        done = self.get_done()
        # split the data into trajectories at the dones
        # add progress key to self.raw_trajectory
        assert(len(progress) == len(self.raw_trajectory))
        #for i in range(len(self.raw_trajectory)):
        #    self.raw_trajectory[i]['progress'] = progress[i]
        split_indices = np.where(done)[0] + 1
        # but also split where we have rapid changes
        # split progress according to the split indices
        #print(split_indices)
        progress = np.split(progress, split_indices)
        #print(len(progress))
        #print(len(progress[1]))
        all_progress = []
        # combine the loops
        for prog_dones in progress:
            # plot progress dones
            if len(prog_dones) == 0:
                break
            delta = np.diff(prog_dones)
            # we need to stich the trajectories together at large deltas
            rapid_changes = np.where(np.abs(delta) > 0.94)[0] +1
            #print(rapid_changes)
            # shift the entire array by the starting index
            new_progress = prog_dones
            for rapid_change in rapid_changes[::-1]:
                new_progress[rapid_change:] += new_progress[rapid_change-1]
            # plot the new progress
            new_progress -= new_progress[0]
            all_progress.append(new_progress)
        
        # scan each progress for rapid changes/ i.e. chrashes and further split there
        progress = []
        for prog in all_progress:
            #print(prog)
            delta = np.diff(prog)
            rapid_changes = np.where(np.abs(delta) > 0.1)[0] +1
            # split at rapid_changes
            new_progress = np.split(prog, rapid_changes)
            #print(new_progress)
            # if new progress is empty, apped prog
            if len(rapid_changes) == 0:
                progress.append(prog)
                continue
            # else append new progress
            else:
                for p in new_progress:

                    progress.append(p)
                    #print(p)

        #plt.title('Split Trajectories on crashes')
        #for prog in progress:
        #    plt.plot(prog)
        #plt.show()
        
        # we now have nicely split trajectories, that are always split on dones and crashes
        # it remains to limit the length of each trajectory to reach at most 1.0
        # loop over all the trajectories
        all_progress = []
        for trajectory in progress:
            # if the trajectory reaches 1.0
            remaining_trajectory = trajectory

            while np.max(remaining_trajectory) >= 1.0:
                #print(len(remaining_trajectory))
                # find where the trajectory reaches 1.0
                one_indices = np.where(remaining_trajectory >= 1.0)[0][0]
                #print(one_indices)
                # split the trajectory at the first 1.0 indices
                new_trajectory = remaining_trajectory[:one_indices+1]
                remaining_trajectory = remaining_trajectory[one_indices+1:]
                # subtract the first value from the remaining trajectory
                remaining_trajectory -= remaining_trajectory[0]
                
                #print(len(remaining_trajectory))
                # append the first X (hopefully less) elements of the new trajectory
                all_progress.append(new_trajectory) # [:FLAGS.max_traj_length])
            all_progress.append(remaining_trajectory)
        # plot 
        #plt.title('Split Trajectories')
        #for prog in all_progress:
        #    plt.plot(prog)
        #plt.show()
        # calulate the overall length of all_progress
        total_length = sum([len(trajectory) for trajectory in all_progress])
        assert(total_length == len(self.raw_trajectory))

        # based on the length of each progress element calulate the spliting points
        split_indices = np.cumsum([len(trajectory) for trajectory in all_progress])[:-1]
        # split the raw_trajectory at the split_indices
        # add progress to raw_trajectory
        flatt_all_progress = np.concatenate(all_progress).flatten()
        for i in range(len(self.raw_trajectory)):
            self.raw_trajectory[i]['progress'] = flatt_all_progress[i]

        self.trajectories = np.split(self.raw_trajectory, split_indices)

        return self.trajectories

    def get_progress(self, centerpoints):
        return self.distance_along_centerline_np(centerpoints, self.pose_points)
    
    def distance_along_centerline_np(self, centerpoints, pose_points):
        centerpoints = np.array(centerpoints)
        pose_points = np.array(pose_points)
        
        # Calculate segment vectors and their magnitudes
        segment_vectors = np.diff(centerpoints, axis=0)
        segment_lengths = np.linalg.norm(segment_vectors, axis=1)
        
        # Extend segment lengths to compute cumulative distance
        cumulative_lengths = np.hstack(([0], np.cumsum(segment_lengths)))
        
        def projected_distance(pose):
            rel_pose = pose - centerpoints[:-1]
            t = np.sum(rel_pose * segment_vectors, axis=1) / np.sum(segment_vectors**2, axis=1)
            t = np.clip(t, 0, 1)
            projections = centerpoints[:-1] + t[:, np.newaxis] * segment_vectors
            distances = np.linalg.norm(pose - projections, axis=1)
            
            closest_idx = np.argmin(distances)
            return cumulative_lengths[closest_idx] + segment_lengths[closest_idx] * t[closest_idx]
        
        return np.array([projected_distance(pose) for pose in pose_points])
    

def getProgress(trajectories):
    # loop over trajectories
    progress = []
    for i in range(len(trajectories)):
        for timestep in trajectories[i]:
            progress.append(timestep['progress'])
    return np.array(progress).flatten()

def getField(trajectories, field):
    # loop over trajectories
    field_list = []
    for i in range(len(trajectories)):
        for timestep in trajectories[i]:
            field_list.append(timestep[field])
    return np.array(field_list).flatten()

def filterTrajectories(trajectories, threshold):
    # remove all trajectories with less than threshold values
    filtered_trajectories = []
    for trajectory in trajectories:
        if len(trajectory) > threshold:
            filtered_trajectories.append(trajectory)
    return filtered_trajectories

def calculateRewardTD(trajectories):
    # calculate time difference reward for each trajectory
    for trajectory in trajectories:
        if len(trajectory) == 0:
            continue
        progress = getProgress([trajectory])
        # do time difference
        reward = np.diff(progress)
        # append an average reward to the end
        reward = np.append(reward, np.mean(reward))
        # set the reward in the trajectory
        for i in range(len(trajectory)):
            trajectory[i]['rewardTD'] = reward[i]
        
    return trajectories

def main(argv):
    # load in the centerline
    centerDs = PickleDataset('middle_line.pkl', skip_loading=True)
    
    x_c, y_c = centerDs.get_pose()
    
    x_c = x_c[634:] #[::-1]
    y_c = y_c[634:] #[::-1]
    x_c[0] = x_c[-1]
    y_c[0] = y_c[-1]
    centerline_points = np.stack((x_c, y_c), axis=-1)

    plt.title('Middle line')
    plt.scatter(x_c, y_c, s=0.2, label='trajectories')
    if FLAGS.plot:
        plt.show()
    
    # check all pickle files available
    all_files = [f for f in os.listdir( FLAGS.input_folder) if os.path.isfile(os.path.join( FLAGS.input_folder, f))]
    print(f"Available files: {all_files}")

    # loop over all the files
    lap_len = lap_length(centerline_points)
    all_trajectories = []
    for file in all_files:
        rawDataset = PickleDataset(file)
        x_pose, y_pose = rawDataset.get_pose()
        plt.scatter(x_c, y_c, s=0.2, label='centerline')
        plt.scatter(x_pose, y_pose, s=0.2, label='trajectories')
        if FLAGS.plot:
            plt.show()
        progress = rawDataset.get_progress(centerpoints=centerline_points)
        progress = progress/ lap_len
        plt.title('Progress')
        plt.plot(progress)
        if FLAGS.plot:
            plt.show()
        
        trajectories = rawDataset.split_trajectories(progress)
        # print(len(trajectories))
        # remove the last trajectory, since it is not complete
        trajectories = trajectories[:-1]
        # filter trajectories
        trajectories = filterTrajectories(trajectories, FLAGS.min_traj_length)


        progress = getProgress(trajectories)
        # print(progress)
        plt.title('Split Trajectories')
        for trajectory in trajectories:
            plt.plot(getProgress([trajectory]))
        if FLAGS.plot:
            plt.show()
        
        # calculate the reward
        trajectories = calculateRewardTD(trajectories)
        # plot the reward for each trajectory
        plt.title('Reward')
        for trajectory in trajectories:
            plt.plot(getField([trajectory], 'rewardTD'))
        if FLAGS.plot:
            plt.show()

        # plot continous reward
        plt.title('Continous Reward')
        plt.plot(getField(trajectories, 'rewardTD'))
        plt.plot(getField(trajectories, 'progress') * 0.005)
        if FLAGS.plot:
            plt.show()


        all_trajectories += trajectories
    # now save trajectories into zarr files
    root = zarr.open('trajectories.zarr', mode='w')
    print(" Length of all trajectories: ", len(all_trajectories))
    for i, trajectory in tqdm(enumerate(all_trajectories)):
        trajectory_group = root.create_group(str(i))

        trajectory_group.array('rewards', data=getField([trajectory],'rewardTD'), chunks=(500,))
        
        progress = getField([trajectory],'progress')
        trajectory_group.array('progress', data=progress, chunks=(500,))
        
        terminals = np.zeros_like(progress)
        terminals[-1] = 1
        trajectory_group.array('terminals', data= terminals, chunks=(500,))

        obs_fields = ['scan', 'pose_x', 'pose_y', 'pose_theta', 'linear_vel_y', 'linear_vel_x', 'ang_vels_z']
        info_fields = ['agent', 'target_speed']
        action_fields = ['steering', 'velocity']
        obs_group = trajectory_group.create_group('observations')
        info_group = trajectory_group.create_group('infos')
        action_group = trajectory_group.create_group('actions')
        
        
        for field in obs_fields:
            if field == 'scan':
                obs_group.zeros(field, shape=(0,1080), chunks=(400,1080), dtype='float32', resizable=True)
            else:
                obs_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)
        
        for field in info_fields:
            if field == 'agent':
                info_group.zeros(field, shape=(0,), chunks=(400,), dtype='str', resizable=True)
            else:
                info_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)

        for field in action_fields:
            action_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)
        # with tqdm

        for j in range(len(trajectory)):
            observation = {
            'scan': np.array(trajectory[j]['scan']),
            'pose_x': trajectory[j]['pose_x'],
            'pose_y': trajectory[j]['pose_y'],
            'pose_theta': trajectory[j]['pose_theta'],
            'linear_vel_y': trajectory[j]['linear_vel_y'],
            'linear_vel_x': trajectory[j]['linear_vel_x'],
            'ang_vels_z': trajectory[j]['ang_vels_z'],
        }
            infos = dict(agent= trajectory[j]['agent'], target_speed= trajectory[j]['target_speed'])
            actions= dict(steering = trajectory[j]['steering'], velocity = trajectory[j]['velocity'])
            for field, value in observation.items():
                # arr = 
                # print(np.squeeze(np.array([value])))
                # print value shape
                obs_group[field].append(np.array([value]))
            for field, value in infos.items():
                # arr = 
                info_group[field].append(np.array([value]))
            for field, value in actions.items():
                # arr = 
                action_group[field].append(np.array([value]))
        # break
    print("Wrote dataset")
    
if __name__ == '__main__':
    app.run(main)
