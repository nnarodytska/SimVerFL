import argparse
import distutils.dir_util


#python3.8 synch_scipts.py --path_from ../xfl/results/federated_cnn/fedavg/ --path_to ./trained_models/fedavg/

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    ### if debug, only prints the command to run, without execution
    parser.add_argument('--path_from', '-p1', required=True, help='path to models')
    parser.add_argument('--path_to', '-p2', required=True, help='path to copy')
    args = parser.parse_args()

    from_dir = args.path_from
    to_dir = args.path_to
    distutils.dir_util.copy_tree(from_dir, to_dir)