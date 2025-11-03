import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
import logging
from config import training_config, lora_config
from utils import seed_torch
from dataset import SequenceDataset
from model import build_model
from evaluate import evaluate_model

def main():
    seed_torch(training_config.SEED)
    logging.basicConfig(filename=training_config.LOG_FILE, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    full_dataset = SequenceDataset(training_config.TRAIN_SEQUENCE_FASTA, training_config.TRAIN_BINDING_SITE_FASTA)
    test_dataset = SequenceDataset(training_config.TEST_SEQUENCE_FASTA, training_config.TEST_BINDING_SITE_FASTA)
    test_loader = DataLoader(test_dataset, batch_size=training_config.BATCH_SIZE, shuffle=False)

    kf = KFold(n_splits=training_config.K_FOLDS, shuffle=True, random_state=training_config.SEED)

    for fold, (train_indices, val_indices) in enumerate(kf.split(full_dataset)):
        logging.info(f"========== Starting Fold {fold+1}/{training_config.K_FOLDS} ==========")
        mlp_config = {'hidden_dims': training_config.MLP_HIDDEN_DIMS, 'dropout': training_config.MLP_DROPOUT}
        attention_config = {'num_layers': training_config.NUM_ATTENTION_LAYERS, 'dropout': training_config.ATTENTION_DROPOUT}
        model, alphabet = build_model(mlp_config, attention_config)
        model.to(training_config.DEVICE)
        
        batch_converter = alphabet.get_batch_converter()
        optimizer = optim.AdamW(model.parameters(), lr=training_config.LEARNING_RATE, weight_decay=training_config.WEIGHT_DECAY)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
        criterion = torch.nn.CrossEntropyLoss()

        train_subset = torch.utils.data.Subset(full_dataset, train_indices)
        val_subset = torch.utils.data.Subset(full_dataset, val_indices)
        train_loader = DataLoader(train_subset, batch_size=training_config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=training_config.BATCH_SIZE, shuffle=False)
        
        for epoch in range(training_config.NUM_EPOCHS):
            model.train()
            running_loss = 0.0
            all_labels_train, all_predictions_train = [], []

            for batch in train_loader:
                sequence = batch['sequence'][0]
                labels = batch['labels'].to(training_config.DEVICE)

                data = [(None, sequence)]
                _, _, batch_tokens = batch_converter(data)
                batch_tokens = batch_tokens.to(training_config.DEVICE)

                optimizer.zero_grad()
                outputs = model(batch_tokens, sequence, training_config.DEVICE)

                outputs_flat = outputs.view(-1, 2)
                labels_flat = labels.view(-1)
                min_len = min(len(outputs_flat), len(labels_flat))
                
                loss = criterion(outputs_flat[:min_len], labels_flat[:min_len])
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                probabilities = F.softmax(outputs_flat[:min_len], dim=1)[:, 1]
                all_labels_train.extend(labels_flat[:min_len].detach().cpu().numpy())
                all_predictions_train.extend(probabilities.detach().cpu().numpy())

            scheduler.step()
            
            avg_loss_train = running_loss / len(train_loader)
            auc_roc_train = roc_auc_score(all_labels_train, all_predictions_train)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1}/{training_config.NUM_EPOCHS} | Train Loss: {avg_loss_train:.4f}, Train AUROC: {auc_roc_train:.3f}")

            val_metrics = evaluate_model(model, val_loader, criterion, batch_converter, training_config.DEVICE)
            logging.info(f"Fold {fold+1}, Epoch {epoch+1} | Val AUROC: {val_metrics['auc_roc']:.3f}, Val AUPRC: {val_metrics['auc_prc']:.3f}")
        
        logging.info("Fold finished. Evaluating on test set.")
        test_metrics = evaluate_model(model, test_loader, criterion, batch_converter, training_config.DEVICE)
        logging.info(f"Fold {fold+1} Test Metrics: {test_metrics}")

    logging.info("Training complete. Saving final model.")
    torch.save(model.state_dict(), training_config.MODEL_SAVE_PATH)


if __name__ == '__main__':
    main()