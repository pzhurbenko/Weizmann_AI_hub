import os
import re
import numpy as np
import pandas as pd
import argparse
import logging
from sklearn import metrics
import matplotlib.pyplot as plt
import xgboost as xgb

import glob
from natsort import natsorted


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-train_emb", type=str, required=True, help="Input path to npz file containing embeddings 'combined_embeddings.npz'")
    parser.add_argument("-valid_emb", type=str, required=True, help="Input path to npz file containing embeddings 'combined_embeddings.npz'")
    parser.add_argument("-test_emb", type=str, help="Input path to npz file containing embeddings 'combined_embeddings.npz'")
    parser.add_argument("-train_labels", type=str, required=True, help="Input path to txt file containing labels 'TAIR10_FT_neutral_vs_simulated_train.txt'")
    parser.add_argument("-valid_labels", type=str, required=True, help="Input path to txt file containing labels")
    parser.add_argument("-test_labels", type=str, help="Input path to txt file containing labels")
    parser.add_argument("-seed", type=int, default=42, help="The random seed to train XGBoost model")
    parser.add_argument("-output", type=str, help="The directory to save output, including XGBoost.json")
    return parser.parse_args()

def train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels, random_state=42):
    logging.info("Training XGBoost model")
    model = xgb.XGBClassifier(n_estimators=1000, max_depth=6, learning_rate=0.1, random_state=random_state, n_jobs=-1)
    model.fit(train_embeddings, train_labels, eval_set=[(valid_embeddings, valid_labels)])
    return model

def infer_xgboost_model(model, embeddings):
    logging.info("Inferencing XGBoost model")
    return model.predict_proba(embeddings)[:, 1]

def evaluate_model(predictions, labels):
    logging.info("Evaluating model")
    fpr, tpr, _ = metrics.roc_curve(labels, predictions)
    precision, recall, _ = metrics.precision_recall_curve(labels, predictions)
    roc_auc = metrics.auc(fpr, tpr)
    prauc = metrics.average_precision_score(labels, predictions)
    return fpr, tpr, precision, recall, roc_auc, prauc

def plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, output_dir, prefix='valid', random_state=42):
    logging.info(f"Plotting metrics and saving to {output_dir}")
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}', linewidth=2)
    axs[0].set_title('ROC Curve')
    axs[0].set_xlabel('False Positive Rate')
    axs[0].set_ylabel('True Positive Rate')
    axs[0].legend(loc='lower right')

    axs[1].plot(recall, precision, label=f'PRAUC = {prauc:.2f}', linewidth=2)
    axs[1].set_title('Precision-Recall Curve')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='lower left')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'seed_{random_state}_{prefix}_metrics.png'))
    # save metrics to file
    with open(os.path.join(output_dir, f'seed_{random_state}_{prefix}_metrics.txt'), 'w') as f:
        f.write(f"ROC AUC: {roc_auc:.2f}\n")
        f.write(f"PRAUC: {prauc:.2f}\n")
        
def load_data(filepath):
    logging.info(f"Loading data from {filepath}")
    data = pd.read_csv(filepath, delimiter='\t')
    return data['sequences'].tolist(), data['label'].tolist()


def main():
    logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
    )
    logging.info(f"Initiating machine learning")

    args = parse_args()
    os.makedirs(args.output, exist_ok=True)

    train_sequences, train_labels = load_data(args.train_labels)
    valid_sequences, valid_labels = load_data(args.valid_labels)
    logging.info(f"Loading embeddings")
    train_embeddings = np.load(args.train_emb)
    valid_embeddings = np.load(args.valid_emb)
    train_embeddings = train_embeddings['train']
    valid_embeddings = valid_embeddings['valid']
                            
    if os.path.exists(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json')):
        logging.info(f"Found pre-trained XGBoost model, loading from file {os.path.join(args.output, f'seed_{args.seed}_XGBoost.json')}")
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
    else:
        xgb_model = train_xgboost_model(train_embeddings, train_labels, valid_embeddings, valid_labels, random_state=args.seed)
        xgb_model.save_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
        valid_predictions = infer_xgboost_model(xgb_model, valid_embeddings)
        np.savez_compressed(os.path.join(args.output, f'seed_{args.seed}_valid_predictions.npz'), predictions=valid_predictions)
        fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(valid_predictions, valid_labels)
        prefix = os.path.basename(args.valid_labels).split('.')[0]
        plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix, random_state=args.seed)

        
    if args.test_labels:
        test_sequences, test_labels = load_data(args.test_labels)
        embeddings = np.load(args.test_emb)
        test_embeddings = embeddings['test']
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(os.path.join(args.output, f'seed_{args.seed}_XGBoost.json'))
        prefix = os.path.basename(args.test_labels).split('.')[0]

        predictions = infer_xgboost_model(xgb_model, test_embeddings)
        np.savez_compressed(os.path.join(args.output, f'seed_{args.seed}_{prefix}_predictions.npz'), predictions=predictions)
        fpr, tpr, precision, recall, roc_auc, prauc = evaluate_model(predictions, test_labels)
        prefix = os.path.basename(args.test_labels).split('.')[0]
        plot_metrics(fpr, tpr, precision, recall, roc_auc, prauc, args.output, prefix=prefix, random_state=args.seed)

if __name__ == "__main__":
    main()
