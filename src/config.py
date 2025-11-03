import torch

class LoRAConfig:
    """LoRA specific configurations."""
    def __init__(self):
        self.lora_rank = 4
        self.lora_init_scale = 0.01
        self.lora_modules = ".*SelfAttention|.*EncDecAttention|.*self_attn|.*fc"
        self.lora_layers = "q|k|v|o|k_proj|v_proj|q_proj|out_proj|fc1|fc2"
        self.trainable_param_names = ".*layer_norm.*|.*lora_[ab].*"
        self.lora_scaling_rank = 1

class TrainingConfig:
    """Training session configurations."""
    def __init__(self):
        self.SEED = 42
        self.DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        
        self.TRAIN_SEQUENCE_FASTA = '/path/to/train_sequences.fasta'
        self.TRAIN_BINDING_SITE_FASTA = '/path/to/train_binding_sites.fasta'
        self.TEST_SEQUENCE_FASTA = '/path/to/test_sequences.fasta'
        self.TEST_BINDING_SITE_FASTA = '/path/to/test_binding_sites.fasta'
        self.MODEL_SAVE_PATH = 'name_of_trained_model.pth'
        self.LOG_FILE = "model_training.log"
        
        
        self.BATCH_SIZE =  32 # e.g., 32
        self.LEARNING_RATE =  0.001 # e.g., 0.001
        self.WEIGHT_DECAY = 1e-5 # e.g., 1e-5
        self.NUM_EPOCHS = 50 # e.g., 50
        self.K_FOLDS = 5 # e.g., 5
        self.MLP_DROPOUT = 0.1 # e.g., 0.1
        self.ATTENTION_DROPOUT = 0.1 # e.g., 0.1
        self.MLP_HIDDEN_DIMS = 256 # e.g., 256
        self.NUM_ATTENTION_LAYERS = 4 # e.g., 4


lora_config = LoRAConfig()
training_config = TrainingConfig()