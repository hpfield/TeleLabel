import argparse
import os
from eval_utils import get_results, print_false_negatives, save_best, plot_metrics, plot_best_thresholds

MULTILABEL_ROOT = os.path.abspath(os.path.join(__file__, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(__file__, '..', '..', '..'))

# Function to set up command line argument parsing
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate machine learning model on specified metrics.')
    parser.add_argument('--data_path', type=str, default=os.path.join(MULTILABEL_ROOT, 'outputs', 'llama3-8B'),
                        help='Path to the data directory')
    parser.add_argument('--results_file', type=str, default=os.path.join(results_dir, 'llama-3-multilabel-classification.csv'),
                        help='Path to save the results CSV file')
    parser.add_argument('--best_results_file', type=str, default=os.path.join(results_dir, 'llama-3-multilabel-classification-best.csv'),
                        help='Path to save the best results CSV file')
    # parser.add_argument('--best_results_plot', type=str, default=os.path.join(results_dir, 'llama-3-multilabel-classification-best.csv'),
    #                     help='Path to save the best results plot')
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all metrics')
    parser.add_argument('-f', '--false_negatives', action='store_true', help='Evaluate false negatives')
    parser.add_argument('-b', '--best', action='store_true', help='Save the best results')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the metrics')
    parser.add_argument('--plot_best_thresholds', action='store_true', help='Plot the best thresholds for each metric')


    return parser

def main(args):
    print(f'Data path: {args.data_path}')
    results_df = get_results(args.data_path)
    results_df.to_csv(args.results_file)

    plots_dir = os.path.join(MULTILABEL_ROOT, 'results', 'llama3-8B', 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    metrics = ['Precision', 'Recall', 'F1_score', 'Accuracy']

    if args.all:
        print_false_negatives(metrics, results_df, args.data_path)
        save_best(metrics, results_df, args.best_results_file)
        plot_metrics(metrics, results_df, plots_dir)
        plot_best_thresholds(args.best_results_file, plots_dir)
    else:
        if args.false_negatives:
            print_false_negatives(metrics, results_df, args.data_path)
        if args.best:
            save_best(metrics, results_df, args.best_results_file)
        if args.plot:
            plot_metrics(metrics, results_df, plots_dir)
        if args.plot_best_thresholds:
            plot_best_thresholds(args.best_results_file, plots_dir)

    print('Evaluation completed.')

if __name__ == "__main__":
    results_dir = os.path.join(MULTILABEL_ROOT, 'results', 'llama3-8B')
    os.makedirs(results_dir, exist_ok=True)
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
