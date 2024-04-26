import pandas as pd
import ast
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, hamming_loss
from sklearn.preprocessing import MultiLabelBinarizer

# Load your data
df = pd.read_csv('/home/rz20505/Documents/ask-jgi/data/labelled/llama-3-telecoms-topics.csv')

# Ensure labels are in list format if they aren't already
df['topics'] = df['topics'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
df['gt'] = df['gt'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

# Evaluation function
def evaluate_predictions(preds, gts):
    total_datapoints = len(gts)
    success_count = 0
    extra_label_count = 0
    
    for pred, gt in zip(preds, gts):
        if set(gt).issubset(pred):
            success_count += 1
            if set(pred) != set(gt): # If not equal, pred must have additional lablel
                extra_label_count += 1
                
    success_rate = success_count / total_datapoints
    extra_label_rate = extra_label_count / total_datapoints
    return success_rate, extra_label_rate

# Evaluation
success_rate, extra_label_rate = evaluate_predictions(df['topics'], df['gt'])

# Print results
print(f"Success Rate: {success_rate:.2%}")
print(f"Extra Label Rate: {extra_label_rate:.2%}")

# MultiLabel Binarizer to handle multi-label classification
mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(df['gt'])
y_pred = mlb.transform(df['topics'])

# Calculating metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='samples')
recall = recall_score(y_true, y_pred, average='samples')
f1 = f1_score(y_true, y_pred, average='samples')
hamming = hamming_loss(y_true, y_pred)  # Measures the fraction of labels that are incorrectly predicted


print(f"Accuracy: {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall: {recall:.2%}")
print(f"F1 Score: {f1:.2%}")
print(f"Hamming Loss: {hamming:.2%}")


# Analysis of Label Specific Performance
label_counts = pd.DataFrame({
    'label': mlb.classes_,
    'false_negatives': (y_true & ~y_pred).sum(axis=0),
    'false_positives': (~y_true & y_pred).sum(axis=0),
    'true_positives': (y_true & y_pred).sum(axis=0),
})

label_counts['miss_rate'] = label_counts['false_negatives'] / (label_counts['false_negatives'] + label_counts['true_positives'])
label_counts['false_positive_rate'] = label_counts['false_positives'] / (label_counts['false_positives'] + label_counts['true_positives'])

print("\nLabel-specific Analysis:")
print(label_counts.sort_values('miss_rate', ascending=False))