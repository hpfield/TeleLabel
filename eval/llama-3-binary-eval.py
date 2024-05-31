import argparse
import os
from eval_utils import get_binary_results, print_false_negatives, save_best, plot_metrics

# Function to set up command line argument parsing
def setup_arg_parser():
    parser = argparse.ArgumentParser(description='Evaluate machine learning model on specified metrics.')
    parser.add_argument('--data_path', type=str, default='data/labelled/llama-3-binary-classification/',
                        help='Path to the data directory')
    parser.add_argument('--results_file', type=str, default='eval/results/llama-3-binary-classification.csv',
                        help='Path to save the results CSV file')
    parser.add_argument('--best_results_file', type=str, default='eval/results/llama-3-binary-classification-best.csv',
                        help='Path to save the best results CSV file')
    parser.add_argument('-a', '--all', action='store_true', help='Evaluate all metrics')
    parser.add_argument('-f', '--false_negatives', action='store_true', help='Evaluate false negatives')
    parser.add_argument('-b', '--best', action='store_true', help='Save the best results')
    parser.add_argument('-p', '--plot', action='store_true', help='Plot the metrics')

    return parser

def main(args):
    os.chdir('..')  # Navigate to the root directory

    results_df = get_binary_results(args.data_path)
    results_df.to_csv(args.results_file)

    metrics = ['Precision', 'Recall', 'F1 Score', 'Accuracy']

    if args.all:
        print_false_negatives(metrics, results_df, args.data_path)
        save_best(metrics, results_df, args.best_results_file)
        plot_metrics(metrics, results_df)
    else:
        if args.false_negatives:
            print_false_negatives(metrics, results_df, args.data_path)
        if args.best:
            save_best(metrics, results_df, args.best_results_file)
        if args.plot:
            plot_metrics(metrics, results_df)

    print('Evaluation completed.')

if __name__ == "__main__":
    parser = setup_arg_parser()
    args = parser.parse_args()
    main(args)
