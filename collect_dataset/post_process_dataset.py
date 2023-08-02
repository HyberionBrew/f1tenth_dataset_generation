# load in dataset, set terminals and compute the rewards
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt # read pickle file from collect_dataset/dataset.pkl
from absl import app, flags
import csv
from scipy.spatial.distance import cdist
import pickle
FLAGS = flags.FLAGS

flags.DEFINE_float('speed', 1.0, 'Mean speed of the car')
flags.DEFINE_string('input_file', 'SFTG_speed_1.0.pkl', 'Name of dataset file')
flags.DEFINE_string('output_file', 'SFTG_speed_1.0_processed.pkl', 'Name of dataset file')
flags.DEFINE_string('centerline_file', 'ellipse_map2.csv', 'Name of centerline file')

def main(argv):
    with open(FLAGS.input_file, 'rb') as f:
        data = []
        while True:
            try:
                data.append(pkl.load(f))
            except EOFError:
                break
    # data is a list of tuples of the form (speed, steering, obs, step_reward, done, info, timestep, dataset_name)
    y_pose = [data[i][2]['poses_y'][0] for i in range(0, len(data))]
    x_pose = [data[i][2]['poses_x'][0] for i in range(0, len(data))]
    # do small points
    # print(len(y_pose))
    # open the centerline
    with open(FLAGS.centerline_file, newline='') as csvfile:
        centerline = list(csv.reader(csvfile))
    
    x_c = [float(centerline[i][0]) for i in range(2, len(centerline))]
    y_c = [float(centerline[i][1]) for i in range(2, len(centerline))]
    # add a point at the end to close the loop. same y as last point and x interpolated between last and first point
    # some manual hack for this specific centerline
    y_c.append(y_c[-1])
    x_c.append((x_c[0]+x_c[-1])/2)
    plt.scatter(x_c, y_c, s=0.2, label='centerline')
    plt.scatter(x_pose, y_pose, s=0.2, label='trajectories')
    plt.show()

    centerline_points = np.stack((x_c, y_c), axis=-1)
    centerline_points = np.flip(centerline_points,axis=0)
    # flip the centerline points to get the correct order
    pose_points = np.stack((x_pose, y_pose), axis=-1)
    distances = cdist(pose_points, centerline_points)
    # print the first 100 pose_points
    #print(pose_points[:100])
    #print("dswwd")
    #print(centerline_points[:100])
    # plot these points
    #plt.scatter(pose_points[:100,0], pose_points[:100,1], s=0.2, label='pose points')
    #plt.scatter(centerline_points[:100,0], centerline_points[:100,1], s=0.2, label='centerline points')
    #plt.legend()
    #plt.show()

    # Get the indices of the closest centerline points
    closest_indices = np.argmin(distances, axis=1)
    reward = closest_indices/len(centerline_points)
    # print the first 100 rewards
    #print(reward[:100])
    plt.title('Reward')
    plt.plot(reward)
    plt.show()

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
        #print(rapid_changes)
        # for each rapid_change continue summing the reward, 
        # i.e. add the last step before the rapid change to the 
        # following rewards
        if start==None:
            start = rapid_changes[0]

        for rapid_change in rapid_changes[::-1]:
            trajectory[rapid_change:] += trajectory[rapid_change-1]
        trajectory -= trajectory[0]
        modified_trajectories.append(trajectory)
    # now make each trajectory have the length of 400
    # if the trajectory reaches 1.0 set all values following 1.0 to 1.0
    # if it does not reach 1.0 set all values following the last value to 0.0
    new_trajectories = []
    
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
    # ensure that each trajectory has at most the length of 400 or reaches at most 1.0
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
    for trajectory in new_trajectories:
        plt.plot(trajectory)
    plt.show()

    #exit(0)
    # split into trajectories at these indices

    trajectories = new_trajectories
    # throw away first and last trajectory, since they are not complete
    trajectories = new_trajectories[1:-1]

    # now plot the trajectories
    for trajectory in trajectories:
        plt.plot(trajectory)
    plt.show()

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

    print("Wrote dataset to", FLAGS.output_file)
    
if __name__ == '__main__':
    app.run(main)
