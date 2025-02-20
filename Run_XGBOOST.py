import os
import numpy as np
import logging
import xgboost as xgb
import argparse

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("-embeddings", type=str, required=True, help="Input path to npz file containing embeddings.npz")
  parser.add_argument("-npz_key", type=str, required=True, help="key:array in NPZ file")
  parser.add_argument("-output_name", type=str, required=True, help="Name of the file, '_predictions.npz' will be added")
  parser.add_argument("-model_path", type=str, required=True, help="The trained model XGBoost.json")
  return parser.parse_args() 

def infer_xgboost_model(model, embeddings):
  logging.info("Inferencing XGBoost model")
  return model.predict_proba(embeddings)[:, 1]

def main():
  logging.basicConfig(
  level=logging.INFO,
  format='%(asctime)s - %(levelname)s - %(message)s',
  datefmt='%Y-%m-%d %H:%M:%S'
  )
  logging.info(f"Initiating machine learning")

  args = parse_args()
  embed = np.load(args.embeddings)
  embeddings = embed[args.npz_key]
  xgb_model = xgb.XGBClassifier()
  xgb_model.load_model(args.model_path)
  directory = os.path.dirname(args.model_path)

  predictions = infer_xgboost_model(xgb_model, embeddings)
  np.savez_compressed(os.path.join(directory, f'{args.output_name}_predictions.npz'), predictions=predictions)
  logging.info(f"Predictions were saved to the {directory}_{args.output_name}_predictions.npz")

if __name__ == "__main__":
  main()
