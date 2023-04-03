import argparse
import multiprocess as mp
from exp import get_experiment
import configs
# yapf: disable
if __name__ == '__main__':
    mp.set_start_method("spawn")

    parser = argparse.ArgumentParser()
    parser.add_argument('--processes',
                        type=int,
                        default=1,
                        help="size of process pool used to run experiments.")
    parser.add_argument('--exp',
                        required=True,
                        type=str,
                        help="name of experiment to run")
    parser.add_argument('--out_dir',
                        required=True,
                        type=str,
                        help="Directory to refer to/write results to")
    parser.add_argument('--result_dir',
                        required=True,
                        type=str,
                        help="Directory to save plots to")
    parser.add_argument('--no_save_out',
                        action='store_false',
                        help="Don't save intermediate results")
    args = parser.parse_args()
    get_experiment(args.exp)(processes=args.processes, out_dir=args.out_dir, result_dir=args.result_dir, save_out=args.no_save_out)
