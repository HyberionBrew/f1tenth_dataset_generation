# load in dataset, set terminals and compute the rewards
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt# read pickle file from collect_dataset/dataset.pkl
from absl import app, flags
import csv
from scipy.spatial.distance import cdist
FLAGS = flags.FLAGS

flags.DEFINE_float('speed', 1.0, 'Mean speed of the car')
flags.DEFINE_string('input_file', 'SFTG_speed_1.pkl', 'Name of dataset file')

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

    # Get the indices of the closest centerline points
    closest_indices = np.argmin(distances, axis=1)
    reward = closest_indices/len(centerline_points)
    plt.plot(reward)
    plt.show()

    # now look at the rapid changes
    d = np.diff(reward)
    rapid_changes = np.where(np.abs(d) > 0.5)[0] +1
    # split into trajectories at these indices
    trajectories = np.split(reward, rapid_changes)
    # throw away first and last trajectory, since they are not complete
    trajectories = trajectories[1:-1]
    # now plot the trajectories
    for trajectory in trajectories:
        plt.plot(trajectory)
    plt.show()

    # now create a dataset in the d4rl format
    dataset = dict(
        actions=[],
        observations=[],
        rewards=[],
        terminals=[])
    

if __name__ == '__main__':
    app.run(main)
