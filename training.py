import os
import json
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import joblib
import random
import gzip
from pathlib import Path

class Config:
    MODEL_TYPE = "bert" # bert or roberta or deberta
    MODEL_NAME = {
        "bert": "bert-base-uncased",
        "roberta": "roberta-base", 
        "deberta": "deberta-v3-base" 
    }[MODEL_TYPE] # Local model path or name
    MAX_LENGTH = 128
    BATCH_SIZE = {
        "bert": 256,
        "roberta": 128,
        "deberta": 32  
    }[MODEL_TYPE]
 
    LEARNING_RATE = {
        "bert": 1e-4,
        "roberta": 6e-5,
        "deberta": 5e-5  
    }[MODEL_TYPE]

    EPOCHS = 10

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SAVE_DIR = {
        "bert": "saved_models/bert_wsd_model",
        "roberta": "saved_models/roberta_wsd_model",
        "deberta": "saved_models/deberta_wsd_model"
    }[MODEL_TYPE]
    os.makedirs(SAVE_DIR, exist_ok=True)
    MODEL_ID = f'{MODEL_NAME}_semcor_unified'

    # RoBERTa-specific optimizer parameters
    WARMUP_RATIO = 0.06
    ADAM_EPSILON = 1e-8
    ADAM_BETA1 = 0.9
    ADAM_BETA2 = {
        "bert": 0.999,
        "roberta": 0.98,
        "deberta": 0.999 
    }[MODEL_TYPE]
    WEIGHT_DECAY = {
        "bert": 0.01,
        "roberta": 0.1,
        "deberta": 0.01
    }[MODEL_TYPE]

    # Model architecture configuration
    NUM_HIDDEN_LAYERS = 12
    HIDDEN_SIZE = 768
    NUM_ATTENTION_HEADS = 12
    INTERMEDIATE_SIZE = 3072

class DeviceBatchManager:
    
    def __init__(self, batch, device):
        self.batch = {k: v.to(device) for k, v in batch.items()}
        self.device = device
    
    def __enter__(self):
        return self.batch
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key in list(self.batch.keys()):
            del self.batch[key]
        torch.cuda.empty_cache()

# data preprocessing
class DataProcessor:
    @staticmethod
    def load_json_data(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def build_label_encoder(data, dropped_class_path=None):
        sense_to_key = {}
        for item in data:
            target = item['polysemous']['name']
            for sense in item['polysemous']['sense_definitions_list']:
            # for sense in item['correct_definitions_in_context']:
                sense_to_key[f"{target}::{sense}"] = 1
        if dropped_class_path:
            with open(dropped_class_path, 'r') as f:
                dropped_classes = json.load(f)
            for item in data:
                if all(i not in dropped_classes for i in item['polysemous']['sense_definitions_list']):
                    senses = item['polysemous']['sense_definitions_list']
                    for sense in senses:
                        if f"{item['polysemous']['name']}::{sense}" not in sense_to_key:
                            sense_to_key[f"{item['polysemous']['name']}::{sense}"] = 1
                            if len(sense_to_key) >= Config.CLASS_NUM:
                                break
        print("Unique senses:", len(sense_to_key))
        le = LabelEncoder()
        le.fit(list(sense_to_key.keys()))
        return le
    
    @staticmethod
    def process_data(data, label_encoder):
        processed = []
        # Filter unknown tags
        filtered_data = [
            item for item in data
            if all(f"{item['polysemous']['name']}::{sense}" in label_encoder.classes_
            for sense in item['correct_definitions_in_context'])
        ]
        for item in tqdm(filtered_data, desc='processing data', position=0):
            context = item['context']
            target = item['target']
            marked_context = context.replace(target, f"[TGT]{target}[/TGT]")
            
            for correct_def in item['correct_definitions_in_context']:
                label_key = f"{item['polysemous']['name']}::{correct_def}"
                if label_key in label_encoder.classes_:
                    processed.append({
                        'input': marked_context,
                        'label': label_encoder.transform([label_key])[0],
                        'target_word': target
                    })
        return processed

# dataset class
class WSDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['input'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# trainer
class WSDTrainer:
    def __init__(self, config):
        self.config = config
        if "roberta" in config.MODEL_NAME.lower():
            self.tokenizer = AutoTokenizer.from_pretrained(
            config.MODEL_NAME, add_prefix_space=True
        )
        else: # bert 和 deberta
            self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        # Add special tokens according to the model type
        if "roberta" in config.MODEL_NAME.lower():
            self.tokenizer.add_tokens([" [TGT]", " [/TGT]"])
        else:
            self.tokenizer.add_tokens(["[TGT]", "[/TGT]"])

    def prepare_data(self, train_path, training_artifacts_path=None, original_data_path='datasets/SemCor/semcor.json'):
        train_data = None
        val_data = None
        if training_artifacts_path:
        # Load saved training parameters
            print(f"Load training artifacts: {training_artifacts_path}")
            train_data, val_data, label_encoder, tokenizer = load_training_artifacts(training_artifacts_path)
            if "roberta" in self.config.MODEL_NAME.lower():
                for item in tqdm(train_data, desc='processing data for roberta', position=0):
                    item['input'] = item['input'].replace("[TGT]", " [TGT]").replace("[/TGT]", " [/TGT]")
                for item in tqdm(val_data, desc='processing data for roberta', position=0):
                    item['input'] = item['input'].replace("[TGT]", " [TGT]").replace("[/TGT]", " [/TGT]")
            else:
                self.tokenizer = tokenizer
        # Load raw data
        if not training_artifacts_path:
            print(f"The training artifact path has not been provided. Reprocess the data, train_data_path: {train_path}")
            train_raw = DataProcessor.load_json_data(train_path)
            val_raw = DataProcessor.load_json_data("datasets/en/semeval2007.json")
        original_raw = DataProcessor.load_json_data(original_data_path)
        # Build a unified label encoder
        if training_artifacts_path:
            self.label_encoder = label_encoder
        else:
            self.label_encoder = DataProcessor.build_label_encoder(original_raw)
        
        if not training_artifacts_path:
            train_keys = set()  # Use a set to remove duplicates
            for item in original_raw:
                for key in item["correct_definition_key"]:
                    train_keys.add(key)
            
            # Filter unknown tags
            filtered_val_raw = []
            for item in val_raw:
                if any(key in train_keys for key in item["correct_definition_key"]):
                    filtered_val_raw.append(item)
            
            # process data
            train_data = DataProcessor.process_data(train_raw, self.label_encoder)
            val_data = DataProcessor.process_data(filtered_val_raw, self.label_encoder)
        
            # save
            save_training_artifacts(train_data, val_data, self.label_encoder, self.tokenizer, f"training_artifacts_{self.config.MODEL_ID}")

        #keyword replacement
        # with open('data\annotations\only_annoted_semcor_key_context.json', 'r') as f:
        #     annotated_data = json.load(f)
        # count = 0
        # for item in train_data:
        #     target_word = item['target_word']
        #     
        #     for annotated_item in annotated_data:
        #         annotated_target = annotated_item['target']
        #         original_context = item['input'].replace("[TGT]", "").replace("[/TGT]", "")
        #         if annotated_target == target_word and annotated_item['context'] == original_context:
        #             item['input'] = annotated_item['key_context'].replace(target_word, f"[TGT]{target_word}[/TGT]")
        #             annotated_data.remove(annotated_item)  # Remove used items to avoid duplicate matches
        #             count += 1
        #             break
        # print(f"Keyword replacement completed, {count} training data entries have been replaced.")

        # create datasets
        train_dataset = WSDDataset(train_data, self.tokenizer, self.config.MAX_LENGTH)
        val_dataset = WSDDataset(val_data, self.tokenizer, self.config.MAX_LENGTH)
        
        # create dataloaders
        self.train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.BATCH_SIZE, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.BATCH_SIZE
        )
        
        # Initialize the model
        if "roberta" in self.config.MODEL_NAME.lower():
            from transformers import RobertaConfig, RobertaForSequenceClassification
            model_config = RobertaConfig.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        elif "deberta" in self.config.MODEL_NAME.lower():
            from transformers import DebertaV2Config, DebertaV2ForSequenceClassification
            
            model_config = DebertaV2Config.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = DebertaV2ForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        else:
            from transformers import BertConfig, BertForSequenceClassification
            model_config = BertConfig.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=len(self.label_encoder.classes_),
                num_hidden_layers=self.config.NUM_HIDDEN_LAYERS,
                hidden_size=self.config.HIDDEN_SIZE,
                num_attention_heads=self.config.NUM_ATTENTION_HEADS,
                intermediate_size=self.config.INTERMEDIATE_SIZE,
            )
            self.model = BertForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                config=model_config,
                ignore_mismatched_sizes=True
            )
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.config.DEVICE)
        
        # Export initial parameter tree
        self.monitor.export_parameter_tree(
            os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_ID}_param_tree.json")
        )
    
    def train(self):
        # Set optimizer parameters according to the model type
        if "roberta" in self.config.MODEL_NAME.lower():
            
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if not any(nd in n for nd in no_decay)],
                        'weight_decay': self.config.WEIGHT_DECAY,
                    },
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                    },
                ]
            optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.LEARNING_RATE,
                    betas=(self.config.ADAM_BETA1, self.config.ADAM_BETA2),
                    eps=self.config.ADAM_EPSILON,
                    weight_decay=self.config.WEIGHT_DECAY
                )
        elif "deberta" in self.config.MODEL_NAME.lower():
            
            no_decay = ['bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if not any(nd in n for nd in no_decay)],
                        'weight_decay': self.config.WEIGHT_DECAY,
                    },
                    {
                        'params': [p for n, p in self.model.named_parameters() 
                                if any(nd in n for nd in no_decay)],
                        'weight_decay': 0.0,
                    },
                ]
            optimizer = AdamW(
                    optimizer_grouped_parameters,
                    lr=self.config.LEARNING_RATE,
                    eps=self.config.ADAM_EPSILON
                )
        else:
            optimizer = AdamW(
                    self.model.parameters(),
                    lr=self.config.LEARNING_RATE,
                    eps=self.config.ADAM_EPSILON
                )
            
        # Learning rate scheduler
        if "roberta" in self.config.MODEL_NAME.lower() or "deberta" in self.config.MODEL_NAME.lower():
            from transformers import get_linear_schedule_with_warmup
            total_steps = len(self.train_loader) * self.config.EPOCHS
            warmup_steps = int(total_steps * self.config.WARMUP_RATIO)  # 6% warmup
            scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=warmup_steps,
                    num_training_steps=total_steps
                )
        else:
            scheduler = None

        best_acc = 0
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_main_loss = 0
            total_layer_losses = [0] * self.config.NUM_HIDDEN_LAYERS

            for step, batch in enumerate(tqdm(self.train_loader, desc=f"Epoch {epoch+1}")):
                with DeviceBatchManager(batch, self.config.DEVICE) as batch_on_device:

                    optimizer.zero_grad()
                    outputs = self.model(**batch_on_device)
                    loss = outputs.loss
                    loss.backward()
                        
                    # gradient clipping
                    if "roberta" in self.config.MODEL_NAME.lower() or "deberta" in self.config.MODEL_NAME.lower():
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    optimizer.step()
                    if scheduler:
                        scheduler.step()
                    total_main_loss += loss.item()
                # Clear the basic tensor
                if 'hidden_states' in locals():
                    del hidden_states
                del outputs
            # eval
            val_acc = self.evaluate()
            avg_loss = total_main_loss / len(self.train_loader)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Val Acc={val_acc:.4f}")
            # # Save the model from each round
            # self.save_model(epoch+1, loss=avg_loss)
            # Only save the model from the last round
            if epoch == self.config.EPOCHS - 1:
                self.save_model('final', loss=avg_loss)
    
    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.config.DEVICE) for k, v in batch.items()}
                outputs = self.model(**batch)
                _, preds = torch.max(outputs.logits, dim=1)
                correct += (preds == batch['labels']).sum().item()
                total += len(batch['labels'])
        
        return correct / total
    
    def save_model(self, epoch, loss):
        save_path = os.path.join(self.config.SAVE_DIR, f"{self.config.MODEL_ID}_{epoch}")
        os.makedirs(save_path, exist_ok=True)
        
        # Ensure that the model is in eval mode
        self.model.eval()
        
        # Save the complete model (including structure and weights)
        self.model.save_pretrained(
            save_path,
            safe_serialization=True
        )
        
        # save tokenizer
        self.tokenizer.save_pretrained(save_path)
        
        # Save label encoder
        joblib.dump(
            self.label_encoder,
            os.path.join(save_path, "label_encoder.joblib"),
            protocol=4
        )
        
        # Save training configuration
        torch.save(
            {
                'epoch': epoch,
                'optimizer_state': self.model.state_dict(),
                'loss': loss,
            },
            os.path.join(save_path, "training_state.pt")
        )
        
        print(f"List of model files: {os.listdir(save_path)}")
        print(f"The model has been saved completely to {save_path}")

def robust_json_save(data, path, compress=True):
    """Safely store JSON containing numeric data"""
    def _converter(o):
        if isinstance(o, (np.integer, np.int64)):
            return int(o)
        elif isinstance(o, (np.float32, np.float64)):
            return float(o)
        elif isinstance(o, torch.Tensor):
            return o.cpu().numpy().tolist()
        elif isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"Unable to serialize type: {type(o)}")
        
    json_str = json.dumps(data, default=_converter, indent=2, ensure_ascii=False)
        
    if compress:
        with gzip.open(path, 'wt', encoding='utf-8') as f:
            f.write(json_str)
    else:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(json_str)

def save_training_artifacts(train_data, val_data, label_encoder, tokenizer, save_dir):
    """Save all training artifacts"""
    save_dir = Path(save_dir)
    os.makedirs(save_dir, exist_ok=True)
        
    # 1. Save main data (JSON)
    robust_json_save(train_data, save_dir/"train_data.json.gz")
    robust_json_save(val_data, save_dir/"val_data.json.gz")
        
    # 2. Save label encoder (Joblib)
    joblib.dump(label_encoder, save_dir/"label_encoder.joblib")
        
    # 3. Save Tokenizer
    tokenizer.save_pretrained(save_dir/"tokenizer")
    
    # 4. Save metadata
    train_metadata = {
        "num_samples": len(train_data),
        "labels": list(label_encoder.classes_),
        "data_schema": {k: type(v).__name__ for k,v in train_data[0].items()}
    }
    with open(save_dir/"metadata.json", 'w') as f:
        json.dump(train_metadata, f, indent=2)
    val_metadata = {
        "num_samples": len(val_data),
        "labels": list(label_encoder.classes_),
        "data_schema": {k: type(v).__name__ for k,v in val_data[0].items()}
    }
    with open(save_dir/"val_metadata.json", 'w') as f:
        json.dump(val_metadata, f, indent=2)

def load_training_artifacts(save_dir):
    """Load all training artifacts"""
    save_dir = Path(save_dir)
        
    # 1. Load main data
    with gzip.open(save_dir/"train_data.json.gz", 'rt', encoding='utf-8') as f:
        train_data = json.load(f)
    with gzip.open(save_dir/"val_data.json.gz", 'rt', encoding='utf-8') as f:
        val_data = json.load(f)
        
    # 2. Load other artifacts
    label_encoder = joblib.load(save_dir/"label_encoder.joblib")
    tokenizer = AutoTokenizer.from_pretrained(save_dir/"tokenizer")
        
    return train_data, val_data, label_encoder, tokenizer
