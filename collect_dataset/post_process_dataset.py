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

FLAGS = flags.FLAGS

flags.DEFINE_float('speed', 1.0, 'Mean speed of the car')
flags.DEFINE_string('input_folder', 'raw_datasets', 'Name of dataset file')
flags.DEFINE_string('output_file', 'SFTG_speed_1.0_processed.pkl', 'Name of dataset file')
flags.DEFINE_string('centerline_file', 'ellipse_map2.csv', 'Name of centerline file')

class PickleDataset(object):
    def __init__(self, path):
        self.data = []
        with open(path, 'rb') as f:
            while True:
                try:
                    self.data.append(pkl.load(f))
                except EOFError:
                    break
    def get_pose(self):
        y_pose = [self.data[i][2]['poses_y'][0] for i in range(0, len(self.data))]
        x_pose = [self.data[i][2]['poses_x'][0] for i in range(0, len(self.data))]
        return x_pose, y_pose
def projection_progress(P, A, B):
    # TODO! this does not work properly (if P,A,B on the line for instance instead of A,P,B)
    """
    Compute the normalized position of the projection of point P 
    onto the line defined by points A and B.
    P, A, B are all 2D points represented as (x, y).
    Returns the progress (a value between 0 and 1).
    """
    # TODO clean up and make numpy proper support for boradcasting

    AP = P - A
    v = B - A

    t = (v[:,None]@v[None,:] /(v @ v.T))  # + A.T

    t = t @ (AP[:,None]) + A[:,None]
    t = t.T[0]
    distA = np.linalg.norm(t - A)
    # distance of t from B
    # distB = np.linalg.norm(t - B)
    # Ensure the progress value is constrained between 0 and 1.
    distAB = np.linalg.norm(B - A)
    val = distA/distAB
    if val <0 or val>1:
        print("hi")
    """
    if val < 0 or val > 1:
        print(A)
        print(B)
        print(P)
        print("------")
        print(distA)
        print(distAB)
        print(val)
        print(t)
        plt.scatter(A[0], A[1])
        plt.scatter(B[0], B[1])
        plt.scatter(P[0], P[1])
        plt.scatter(t[0], t[1])
        # add legend
        plt.legend(['A', 'B', 'P', 't'])
        plt.show()
    """
    # cap val between 0 and 1
    val = np.clip(val, 0, 0.5)
    # assert(val >= 0 and val <= 1)
    return val

def distance_from_point_to_line(pose, point1, point2):
    x0, y0 = pose[:,0], pose[:,1]
    x1, y1 = point1[:,0], point1[:,1]
    x2, y2 = point2[:,0], point2[:,1]
    print(x0.shape)
    print(x1.shape)
    print(x2.shape)
    numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denominator = np.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return numerator / denominator


def progress_along_centerline(pose, centerline):
    """
    Computes the progress of the car's pose along a centerline made up of 
    consecutive points.
    
    Args:
    - pose (tuple): A tuple (x, y) representing the car's position.
    - centerline (array): An Nx2 array where N is the number of points 
                          defining the centerline.

    Returns:
    - progress (float): A number between 0 and 1 indicating the car's progress 
                        along the centerline.
    """
    def projection_point_on_segment(P, A, B):
        AP = P - A
        AB = B - A
        t = np.dot(AP, AB) / np.dot(AB, AB)
        return A + t * AB

    P = np.array(pose)

    # Calculate the projection points on each segment and their distances
    projection_points = np.array([projection_point_on_segment(P, centerline[i], centerline[i+1]) for i in range(len(centerline)-1)])
    distances = np.linalg.norm(projection_points - P, axis=1)

    # Find the segment with the closest projection point
    closest_segment_index = np.argmin(distances)

    # Calculate progress along that segment
    segment_start = centerline[closest_segment_index]
    segment_end = centerline[closest_segment_index + 1]
    segment_length = np.linalg.norm(segment_end - segment_start)
    segment_progress = np.linalg.norm(projection_points[closest_segment_index] - segment_start) / segment_length

    # Calculate overall progress along the centerline
    total_length_before_segment = sum([np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(closest_segment_index)])
    total_length = total_length_before_segment + segment_progress * segment_length
    centerline_length = sum([np.linalg.norm(centerline[i+1] - centerline[i]) for i in range(len(centerline)-1)])

    progress = total_length / centerline_length
    return progress

def segment_length(p1, p2):
    """Compute the distance between two points."""
    return np.linalg.norm(p1 - p2)

def point_to_segment_distance(point, start, end):
    """Compute the distance of a point to a segment and the closest point on the segment."""
    segment_vector = end - start
    point_vector = point - start
    t = np.dot(point_vector, segment_vector) / np.dot(segment_vector, segment_vector)
    
    if t < 0.0:
        closest_point = start
    elif t > 1.0:
        closest_point = end
    else:
        closest_point = start + t * segment_vector
    
    distance = np.linalg.norm(point - closest_point)
    return distance, closest_point

def lap_length(centerpoints):
    centerpoints = np.array(centerpoints)
    differences = np.diff(centerpoints, axis=0)
    segment_lengths = np.linalg.norm(differences, axis=1)
    return segment_lengths.sum()


def distance_along_centerline(centerpoints, pose_points):
    centerpoints = np.array(centerpoints)
    pose_points = np.array(pose_points)
    
    distances_along = []

    for pose in pose_points:
        min_distance = float('inf')
        cumulative_distance = 0.0
        projected_cumulative_distance = 0.0
        
        for i in range(len(centerpoints) - 1):
            start, end = centerpoints[i], centerpoints[i+1]
            distance, closest_point_on_segment = point_to_segment_distance(pose, start, end)
            
            if distance < min_distance:
                min_distance = distance
                projected_cumulative_distance = cumulative_distance + segment_length(start, closest_point_on_segment)
                
            cumulative_distance += segment_length(start, end)
            
        distances_along.append(projected_cumulative_distance)
        
    return distances_along

def distance_along_centerline_np(centerpoints, pose_points):
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

import os
def main(argv):
    # load in the centerline
    centerDs = PickleDataset('middle_line.pkl')
    x_c, y_c = centerDs.get_pose()
    plt.title('Middle line')
    plt.scatter(x_c[634:], y_c[634:], s=0.2, label='trajectories')
    plt.show()
    
    # check all pickle files available
    all_files = [f for f in os.listdir( FLAGS.input_folder) if os.path.isfile(os.path.join( FLAGS.input_folder, f))]
    print(f"Available files: {all_files}")

    # loop over all the files
    for file in all_files:
        PickleDataset(file)
        
    with open(FLAGS.input_file, 'rb') as f:
        while True:
            try:
                data.append(pkl.load(f))
            except EOFError:
                break
    y_pose = [data[i][2]['poses_y'][0] for i in range(0, len(data))]
    x_pose = [data[i][2]['poses_x'][0] for i in range(0, len(data))]


    x_c = x_c[634:][::-1]
    y_c = y_c[634:][::-1]
    x_c[0] = x_c[-1]
    y_c[0] = y_c[-1]

    #print(x_c)
    #print(x_pose)
    plt.scatter(x_c, y_c, s=0.2, label='centerline')
    plt.scatter(x_pose, y_pose, s=0.2, label='trajectories')
    plt.show()
    centerline_points = np.stack((x_c, y_c), axis=-1)
    centerline_points = np.flip(centerline_points,axis=0)
    # flip the centerline points to get the correct order
    pose_points = np.stack((x_pose, y_pose), axis=-1)
    distances = cdist(pose_points, centerline_points)


    #closest_indices = np.argmin(distances, axis=1)
    closest_indices = np.argsort(distances, axis=1)[:, :2]
    #for i in range(pose_points.shape[0]):
    raw_reward = closest_indices[:,0]
    # plot the raw reward
    plt.title('Raw reward')
    plt.plot(raw_reward)
    plt.show()
    progress = np.zeros((len(raw_reward)))
    # progress = distance_along_centerline(centerline_points[::5], pose_points) 
    progress = distance_along_centerline_np(centerline_points, pose_points)
    progress = progress/ lap_length(centerline_points)
    # reward = closest_indices/len(centerline_points)
    # print the first 100 rewards
    #print(reward[:100])
    plt.title('Reward')
    plt.plot(progress)
    plt.show()
    reward = progress

    # plot the first 100 dones
    done = [data[i][4] for i in range(0, len(data))]
    # split the data into trajectories at the dones
    trajectories = np.split(reward, np.where(done)[0]+1)[0:-1]
    # in each trajectorie if there is a rapid change in reward add 
    modified_trajectories = []
    #print(len(trajectories[0]))
    start = None
    for trajectory in trajectories:
        # subtract the inital reward from the trajectory

        d = np.diff(trajectory)
        rapid_changes = np.where(np.abs(d) > 0.5)[0] +1
        # plot d
        plt.title('d')
        plt.plot(d)
        plt.show()
        print(rapid_changes)
        # for each rapid_change continue summing the reward, 
        # i.e. add the last step before the rapid change to the 
        # following rewards
        if start==None:
            start = rapid_changes[0]

        for rapid_change in rapid_changes[::-1]:
            trajectory[rapid_change:] += trajectory[rapid_change-1]
        trajectory -= trajectory[0]
        modified_trajectories.append(trajectory)
    new_trajectories = []
    # plot modified trajectories
    plt.title('Modified trajectories')
    for trajectory in modified_trajectories:
        plt.plot(trajectory)
    plt.show()
    """
    for trajectory in modified_trajectories:
        new_trajectory = np.zeros(400)
        new_trajectory[:len(trajectory)] = trajectory
        if trajectory[-1] > 1.0:
            # find where the trajectory reaches 1.0
            one_indices = np.where(trajectory > 1.0)[0]
            # set all values following the first one index to 1.0
            new_trajectory[one_indices[0]:] = 1.0
        else:
            # set all values following the last index to 0.0
            new_trajectory[len(trajectory):] = 0.0
        new_trajectories.append(new_trajectory)
    """
    # ensure that each trajectory has at most the length of reaches at most 1.0
    for trajectory in modified_trajectories:
        # if max in trajectory > 1.0
        remaining_trajectory = trajectory
        while len(remaining_trajectory) > 3:
            # if the trajectory reaches 1.0

            #print(remaining_trajectory)
            if np.max(remaining_trajectory) <= 1.0:
                new_trajectories.append(remaining_trajectory)
                break
            one_indices = np.where(remaining_trajectory >= 1.0)[0][0]
            # split trajectory at the first one index
            #print(one_indices)
            new_trajectories.append(remaining_trajectory[:one_indices+1])
            zero = remaining_trajectory[one_indices+1]
            #print(zero)
            remaining_trajectory = remaining_trajectory[one_indices+1:] - zero

    #new_trajectories = modified_trajectories
    # plot the trajectories
    plt.title('Trajectories')
    for trajectory in new_trajectories:
        plt.plot(trajectory)
    plt.show()

    #exit(0)
    # split into trajectories at these indices

    trajectories = new_trajectories
    # throw away first and last trajectory, since they are not complete
    trajectories = new_trajectories[1:-1]

    # now plot the trajectories

    # calculate time difference reward for each trajectory
    for i in range(len(trajectories)):
        trajectories[i] = np.diff(trajectories[i])
        trajectories[i] = np.append(trajectories[i], 0.0)
    
    plt.title('Time diff reward')
    for trajectory in trajectories:
        plt.plot(trajectory)
    plt.show()
    # now save into zarr files
    root = zarr.open('trajectories.zarr', mode='w')

    # get length of all trajectories numpy
    total_length = sum([len(trajectory) for trajectory in trajectories])
    for i, trajectory in enumerate(trajectories):
        trajectory_group = root.create_group(str(i))
        trajectory_group.array('rewards', data=trajectory, chunks=(400,))
        terminals = np.zeros_like(trajectory)
        terminals[-1] = 1
        
        trajectory_group.array('terminals', data= terminals, chunks=(400,))
        fields_obs = ['scan', 'pose_x', 'pose_y', 'pose_theta', 'linear_vel_y', 'linear_vel_x', 'ang_vels_z']
        obs_group = trajectory_group.create_group('observations')
        info_group = trajectory_group.create_group('infos')
        action_group = trajectory_group.create_group('actions')
        fields_info = ['agent', 'target_speed']
        fields_action = ['steering', 'velocity']
        
        for field in fields_obs:
            if field == 'scan':
                obs_group.zeros(field, shape=(0,1080), chunks=(400,1080), dtype='float32', resizable=True)
            else:
                obs_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)
        
        for field in fields_info:
            if field == 'agent':
                info_group.zeros(field, shape=(0,), chunks=(400,), dtype='str', resizable=True)
            else:
                info_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)

        for field in fields_action:
            action_group.zeros(field, shape=(0,), chunks=(400,), dtype='float32', resizable=True)

        for i in range(start, total_length+start):
            observation = {
            'scan': np.array(data[i][2]['scans'][0]),
            'pose_x': data[i][2]['poses_x'][0],
            'pose_y': data[i][2]['poses_y'][0],
            'pose_theta': data[i][2]['poses_theta'][0],
            'linear_vel_y': data[i][2]['linear_vels_y'][0],
            'linear_vel_x': data[i][2]['linear_vels_x'][0],
            'ang_vels_z': data[i][2]['ang_vels_z'][0],
        }
            infos = dict(agent=data[i][-1][0], target_speed=data[i][-1][1])
            actions= dict(steering = data[i][1], velocity = data[i][0])
        
        for field, value in observation.items():
            arr = obs_group[field]
            # print(np.squeeze(np.array([value])))
            # print value shape
            arr.append(np.array([value]))
        for field, value in infos.items():
            arr = info_group[field]
            arr.append(np.array([value]))
        for field, value in actions.items():
            arr = action_group[field]
            arr.append(np.array([value]))
        



    """
    # now create a dataset in the d4rl format
    dataset = dict(
        actions=[],
        observations=[],
        rewards=[],
        terminals=[],
        infos=[],)
    # split the 
    # paste the rewards into the dataset
    dataset['rewards'] = np.concatenate(trajectories)
    # set the terminals 
    dataset['terminals'] = np.zeros_like(dataset['rewards'])
    # the terminals are 1 at the end of each trajectory
    offset = 0
    for trajectory in trajectories:
        dataset['terminals'][offset + len(trajectory)-1] = 1
        offset += len(trajectory)
    # set the observations
    for i in range(start, len(dataset['rewards'])+start):
        dataset["observations"].append(dict(scan = data[i][2]['scans'][0],
                                            pose_x = data[i][2]['poses_x'][0],
                                            pose_y = data[i][2]['poses_y'][0],
                                            pose_theta = data[i][2]['poses_theta'][0],
                                            linear_vel_y = data[i][2]['linear_vels_y'][0],
                                            linear_vel_x = data[i][2]['linear_vels_x'][0],
                                            ang_vels_z = data[i][2]['ang_vels_z'][0],))
        dataset['infos'].append(dict(agent=data[i][-1][0],
                                speed = data[i][-1][1]))
        dataset['actions'].append((data[i][1], data[i][0])) # we write steering, velocity
    # the observations 
    # plot the rewards

   
    plt.plot(dataset['rewards'])
    plt.plot(dataset['terminals'])
    plt.show()

    x = [dataset['observations'][i]['pose_x'] for i in range(0, len(dataset['observations']))]
    y = [dataset['observations'][i]['pose_y'] for i in range(0, len(dataset['observations']))]
    #plt.scatter(x,y, s=0.2)
    plt.plot(x)
    plt.plot(dataset['rewards'])
    plt.plot(dataset['terminals'])
    plt.show()
    # save the dataset to a file
    with open(FLAGS.output_file, 'wb') as f:
        pickle.dump(dataset, f)
    """
    print("Wrote dataset to", FLAGS.output_file)
    
if __name__ == '__main__':
    app.run(main)
