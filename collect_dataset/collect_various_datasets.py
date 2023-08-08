import subprocess

def run_collect_dataset(agent, speed):
    # Construct the command
    cmd = [
        'python', 'collect_dataset.py',
        '--timesteps=100000',
        '--record',
        '--norender',
        f'--agent={agent}',
        f'--dataset_name={agent}_{speed}.pkl',
        f'--speed={speed}'
    ]

    # Call the command using subprocess
    subprocess.run(cmd)


def main():
    # Utilizing StochasticFTGAgent with the speed values
    for speed in [1.0, 2.0, 3.0, 4.0, 5.0]:
        run_collect_dataset('StochasticFTGAgent', speed)

    # Utilizing StochasticFTGAgentDynamicSpeed with the speed values
    for speed in [0.5, 1.0, 2.0, 3.0, 4.0]:
        run_collect_dataset('StochasticFTGAgentDynamicSpeed', speed)

    cmd = ['python', 'post_process_dataset.py', '--noplot', '--save_figs']
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
