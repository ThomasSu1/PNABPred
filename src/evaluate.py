import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    precision_score, recall_score, f1_score, jaccard_score, 
    matthews_corrcoef, balanced_accuracy_score, confusion_matrix
)
from torch.utils.data import DataLoader
import logging
from config import training_config
from dataset import SequenceDataset
from model import build_model

def evaluate_model(model, loader, criterion, batch_converter, device):
    """Evaluates the model on a given data loader and returns a dictionary of metrics."""
    model.eval()
    total_loss = 0.0
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for batch in loader:
            sequence = batch['sequence'][0]
            labels = batch['labels'].to(device)
            
            data = [(None, sequence)]
            _, _, batch_tokens = batch_converter(data)
            batch_tokens = batch_tokens.to(device)

            outputs = model(batch_tokens, sequence, device)
            outputs_flat = outputs.view(-1, 2)
            labels_flat = labels.view(-1)
            
            min_len = min(len(outputs_flat), len(labels_flat))
            outputs_flat = outputs_flat[:min_len]
            labels_flat = labels_flat[:min_len]
            
            loss = criterion(outputs_flat, labels_flat)
            total_loss += loss.item()

            probabilities = F.softmax(outputs_flat, dim=1)[:, 1]
            all_labels.extend(labels_flat.cpu().numpy())
            all_predictions.extend(probabilities.cpu().numpy())

    all_labels = np.array(all_labels)
    all_predictions_prob = np.array(all_predictions)
    all_predictions_class = (all_predictions_prob > 0.5).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(all_labels, all_predictions_class).ravel()
    
    metrics = {
        "avg_loss": total_loss / len(loader),
        "auc_roc": roc_auc_score(all_labels, all_predictions_prob),
        "auc_prc": average_precision_score(all_labels, all_predictions_prob),
        "accuracy": accuracy_score(all_labels, all_predictions_class),
        "precision": precision_score(all_labels, all_predictions_class, zero_division=0),
        "recall": recall_score(all_labels, all_predictions_class, zero_division=0),
        "f1_score": f1_score(all_labels, all_predictions_class, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "mcc": matthews_corrcoef(all_labels, all_predictions_class),
    }
    return {k: round(v, 4) for k, v in metrics.items()}


def main():
    logging.info("Starting standalone evaluation...")

    MODEL_PATH = '/path/to/pretrained/model.pth'
    mlp_config = {'hidden_dims': training_config.MLP_HIDDEN_DIMS, 'dropout': training_config.MLP_DROPOUT}
    attention_config = {'num_layers': training_config.NUM_ATTENTION_LAYERS, 'dropout': training_config.ATTENTION_DROPOUT}
    model, alphabet = build_model(mlp_config, attention_config)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=training_config.DEVICE))
    model.to(training_config.DEVICE)

    batch_converter = alphabet.get_batch_converter()
    criterion = torch.nn.CrossEntropyLoss()

    test_dataset = SequenceDataset(training_config.TEST_SEQUENCE_FASTA, training_config.TEST_BINDING_SITE_FASTA)
    test_loader = DataLoader(test_dataset, batch_size=training_config.BATCH_SIZE, shuffle=False)

    test_metrics = evaluate_model(model, test_loader, criterion, batch_converter, training_config.DEVICE)
    logging.info(f'Standalone Test Metrics: {test_metrics}')
    print(f'Standalone Test Metrics: {test_metrics}')


if __name__ == '__main__':
    main()