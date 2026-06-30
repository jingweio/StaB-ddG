import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForMaskedLM
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType
)

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY' 

class SigmoidScaling(nn.Module):
    def __init__(self, min_val=-1, max_val=5, initial_scaling=0.8):
        super(SigmoidScaling, self).__init__()
        self.min_val = min_val
        self.max_val = max_val
        
        self.alpha1 = nn.Parameter(torch.tensor(initial_scaling)) 
        self.alpha2 = nn.Parameter(torch.tensor(initial_scaling)) 
        
        self.shift1 = 4
        self.shift2 = 0

    
    def forward(self, x):
        output = torch.zeros_like(x)
        condition1 = x < 0
        output[condition1] = 2 / (1 + torch.exp(-self.alpha2 * (x[condition1] + self.shift2))) - 1

        condition2 = (x >= 0) & (x <= 4)
        output[condition2] = x[condition2]
    
        condition3 = x > 4
        output[condition3] = 2 / (1 + torch.exp(-self.alpha1 * (x[condition3] - self.shift1))) + 3
        
        return output

    
class Stability_classification_head(nn.Module):
    def __init__(self, input_dim=1280, output_dim=10):
        super(Stability_classification_head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)


class ESM3_Stability_head(nn.Module):
    """Stability head for ESM3 — same as Stability_classification_head but with LayerNorm."""
    def __init__(self, input_dim=1536, output_dim=1):
        super(ESM3_Stability_head, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.classifier(x)
        
class ModelWrapper(nn.Module):
    def __init__(self, base_model, stability_head, device):
        super(ModelWrapper, self).__init__()
        self.base_model = base_model
        self.stability_head = stability_head.to(device)


    def forward(self, S):
        # Get the base model outputs
        hidden_dim, mask = self.base_model(S)
        
     
        stability_output = self.stability_head(hidden_dim)
        return stability_output, mask


class FinetuneSaProtModel(nn.Module):
    def __init__(self, model_name, freeze_weights=False):
        super(FinetuneSaProtModel, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name).to(self.device)
        self.freeze_weights = freeze_weights
        
        if self.freeze_weights:
            self._freeze_model_weights()

    def _freeze_model_weights(self):
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, sentences):
        inputs = self.tokenizer(sentences, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        input_ids = inputs["input_ids"]
        mask = torch.where((input_ids == 0) | (input_ids == 1) | (input_ids == 2), 
                           torch.tensor(0, device=self.device), 
                           torch.tensor(1, device=self.device))
        
        
        outputs = self.model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states 
        
        return hidden_states[-1], mask


class TransferModel(nn.Module):
    def __init__(self, cfg, base_model_path=None, additional_layers_path=None):
        super().__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.cfg = cfg
        self.rank = cfg.training.rank if 'rank' in cfg.training else 4
        self.dropout = cfg.training.dropout if 'dropout' in cfg.training else 0.15
        model_name = "westlake-repl/SaProt_650M_AF2"
        
        # Initialize base SaProt model
        self.saprot = FinetuneSaProtModel(model_name, freeze_weights=cfg.model.freeze_weights)

        config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            target_modules=["query", "key", "value", "intermediate.dense", "output.dense"],
            r=self.rank,
            lora_dropout=self.dropout
        )
        
        _ = get_peft_model(self.saprot, config)
        
        # Initialize additional layers
        self.stability_head = Stability_classification_head(input_dim=1280, output_dim=1)
        self.saprot_stability_model = ModelWrapper(self.saprot, self.stability_head, self.device)
            
        if hasattr(cfg, 'training') and hasattr(cfg.training, 'initial_scaling'):
            self.output_scaling = SigmoidScaling(
                min_val=-1, 
                max_val=5, 
                initial_scaling=cfg.training.initial_scaling
            )
        else:
            self.output_scaling = SigmoidScaling(
                min_val=-1, 
                max_val=5
            )
    
        if hasattr(cfg, 'testing') and cfg.testing.ddg_scanning:
            self.ddg_scanning = True
        else:
            self.ddg_scanning = False
            
        # Load weights if provided
        if base_model_path:
            self.load_base_model(base_model_path)
        if additional_layers_path:
            self.load_additional_layers(additional_layers_path)
            
    def load_base_model(self, checkpoint_path):
        """Load only the base SaProt + LoRA model weights"""
        print(f"Loading base model from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract base model state dict
        base_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if any(prefix in key for prefix in ['saprot.', 'base_model.']):
                # Remove the prefix to match current model structure
                new_key = key.replace('model.', '')  # Remove model prefix if present
                base_state_dict[new_key] = value
        
        # Load base model weights
        missing_keys, unexpected_keys = self.saprot.load_state_dict(base_state_dict, strict=False)
        print(f"Base model loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        
    def load_additional_layers(self, checkpoint_path):
        """Load only the additional layers (stability head + output scaling) weights"""
        print(f"Loading additional layers from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract additional layers state dict
        additional_state_dict = {}
        for key, value in checkpoint['state_dict'].items():
            if any(prefix in key for prefix in ['stability_head.', 'output_scaling.', 'saprot_stability_model.']):
                # Remove the prefix to match current model structure
                new_key = key.replace('model.', '')  # Remove model prefix if present
                additional_state_dict[new_key] = value
        
        # Load additional layers weights
        missing_keys, unexpected_keys = self.load_state_dict(additional_state_dict, strict=False)
        print(f"Additional layers loaded. Missing keys: {len(missing_keys)}, Unexpected keys: {len(unexpected_keys)}")
        
    def save_base_model(self, save_path):
        """Save only the base SaProt + LoRA model weights"""
        base_state_dict = {}
        for key, value in self.saprot.state_dict().items():
            base_state_dict[f'model.saprot.{key}'] = value
            
        torch.save({
            'state_dict': base_state_dict,
            'config': self.cfg
        }, save_path)
        print(f"Base model saved to: {save_path}")
        
    def save_additional_layers(self, save_path):
        """Save only the additional layers weights"""
        additional_state_dict = {}
        
        # Save stability head
        for key, value in self.stability_head.state_dict().items():
            additional_state_dict[f'model.stability_head.{key}'] = value
            
        # Save output scaling
        for key, value in self.output_scaling.state_dict().items():
            additional_state_dict[f'model.output_scaling.{key}'] = value
            
        torch.save({
            'state_dict': additional_state_dict,
            'config': self.cfg
        }, save_path)
        print(f"Additional layers saved to: {save_path}")
            
    def forward(self, S):
        if self.ddg_scanning == False:
            stability_output, mask = self.saprot_stability_model(S)
            dg = stability_output.squeeze(-1)
            scaled_dg = self.output_scaling(dg)
            return dg, scaled_dg, mask
    

        elif self.ddg_scanning:
            length = int(len(S[0])/2)
            ddg_scan = torch.zeros(length, 20, len(S), length)
            scaled_ddg_scan = torch.zeros(length, 20, len(S), length)
            stability_output, mask = self.saprot_stability_model(S)
            dg_wt = stability_output.squeeze(-1)
            scaled_dg_wt = self.output_scaling(dg_wt)
            
            for i in range(length):
                for j, A in enumerate(list(ALPHABET)):
                    MUT = list(S[0])
                    MUT[i*2] = A
                    MUT= "".join(MUT)
                    stability_output, mask = self.saprot_stability_model(MUT)
                    dg_mut = stability_output.squeeze(-1)
                    scaled_dg_mut = self.output_scaling(dg_mut)
                    ddg_scan[i][j] = (dg_mut-dg_wt)[:, 1:-1]
                    scaled_ddg_scan[i][j] = (scaled_dg_mut-scaled_dg_wt)[:, 1:-1]
            return ddg_scan, scaled_ddg_scan 