import argparse
import os
from eval_utils import get_results, save_best, plot_metrics
import json
from datetime import datetime

BINARY_ROOT = os.path.abspath(os.path.join(__file__, '../../'))


def get_latest_json_file(directory):
    # Get all json files in the directory
    json_files = [f for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found in the directory.")
        return None
    
    # Parse the timestamp from the filename and find the latest one
    latest_file = max(json_files, key=lambda f: datetime.strptime(f.split('.')[0], '%Y-%m-%d_%H-%M-%S'))
    
    # Construct the full path to the latest file
    latest_file_path = os.path.join(directory, latest_file)
    
    return latest_file_path


# Function to set up command line argument parsing
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate machine learning model on specified metrics.')
    parser.add_argument('--data_path', type=str, default=get_latest_json_file(os.path.join(BINARY_ROOT, 'outputs', 'llama3-8B')),
                        help='Path to the data directory')
    parser.add_argument('--results_file', type=str, default=os.path.join(BINARY_ROOT, 'results', 'llama3-8B', 'threshold_sweep.csv'),
                        help='Path to save the results CSV file')
    parser.add_argument('--best_results_file', type=str, default=os.path.join(BINARY_ROOT, 'results', 'llama3-8B', 'best.csv'),
                        help='Path to save the best results CSV file')
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all metrics')
    parser.add_argument('-b', '--best', action='store_true', help='Save the best results')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the metrics')

    return parser

def main(args):

    results_df = get_results(args.data_path)
    print(results_df.head())
    results_df.to_csv(args.results_file)

    metrics = ['Precision', 'Recall', 'F1_score', 'Accuracy']

    plots_dir = os.path.join(BINARY_ROOT, 'eval', 'plots', 'llama3-8B')

    if args.all:
        save_best(metrics, results_df, args.best_results_file)
        plot_metrics(metrics, results_df, plots_dir)
    else:
        if args.best:
            save_best(metrics, results_df, args.best_results_file)
        if args.plot:
            plot_metrics(metrics, results_df, plots_dir)

    print('Evaluation completed.')

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
