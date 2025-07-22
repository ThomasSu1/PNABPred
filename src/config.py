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
        
        
        self.TRAIN_SEQUENCE_FASTA = '/data/szd22/DNA/DNA573/573S.fasta'
        self.TRAIN_BINDING_SITE_FASTA = '/data/szd22/DNA/DNA573/573L.fasta'
        self.TEST_SEQUENCE_FASTA = '/data/szd22/DNA/DNA573/129S.fasta'
        self.TEST_BINDING_SITE_FASTA = '/data/szd22/DNA/DNA573/129L.fasta'
        self.MODEL_SAVE_PATH = 'dbs_with_esm_lora_mlp_model.pth'
        self.LOG_FILE = "model_training.log"
        
        
        self.BATCH_SIZE =  
        self.LEARNING_RATE =  
        self.WEIGHT_DECAY =  
        self.NUM_EPOCHS =  
        self.K_FOLDS =  
        self.MLP_DROPOUT =  
        self.ATTENTION_DROPOUT =  
        self.MLP_HIDDEN_DIMS =  
        self.NUM_ATTENTION_LAYERS =  


lora_config = LoRAConfig()
training_config = TrainingConfig()