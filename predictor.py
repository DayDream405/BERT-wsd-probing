import torch
from train_monitor import OptimizedModuleSelector
from training import DataProcessor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import joblib
import os
import numpy as np
import json
from tqdm import tqdm
from typing import List
from collections import defaultdict

class WSDPredictor:
    def __init__(self, model_path="./saved_models/best_model"):
        self.path = model_path
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # load model
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # load label encoder
        self.label_encoder = joblib.load(os.path.join(model_path, "label_encoder.joblib"))
    
    def predict_with_attention(self, context, target_word, return_attention=True):
        """
        Predict and return attention scores
        
        Args:
            context: contextual text
            target_word: target word
            return_attention: whether to return the attention score
        
        Returns:
            Dictionary containing prediction results and attention scores
        """
        # mark the target word
        if "roberta" in str(self.model.config.model_type).lower():
            marked_context = context.replace(target_word, f" [TGT]{target_word}[/TGT]")
        else:
            marked_context = context.replace(target_word, f"[TGT]{target_word}[/TGT]")
        
        # encoding
        encoding = self.tokenizer(
            marked_context,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # pred
        with torch.no_grad():
            outputs = self.model(
                **encoding,
                output_attentions=return_attention,
                output_hidden_states=True
            )
            
            logits = outputs.logits
            pred_id = torch.argmax(logits).item()
            pred_label = self.label_encoder.inverse_transform([pred_id])[0]
        
        target, definition = pred_label.split("::", 1)
        confidence = torch.softmax(logits, dim=1).max().item()

        # get probs
        all_probs = {}
        probabilities = torch.softmax(logits, dim=1)[0]
        for i, class_name in enumerate(self.label_encoder.classes_):
            all_probs[class_name] = probabilities[i].item()
        
        result = {
            'target_word': target,
            'predicted_label': pred_label,
            'predicted_definition': definition,
            'confidence': confidence,
            'context': context,
            'all_probabilities': all_probs
        }

        # add attentnion scores
        if return_attention and hasattr(outputs, 'attentions'):
            attentions = self._process_attention_scores(
                outputs.attentions, 
                encoding, 
                target_word
            )
            result['attention_scores'] = attentions
        
        if 'aggregated_attention' in attentions:
            result['aggregated_attention_summary'] = {
                'top_tokens': attentions['aggregated_attention']['top_attended_tokens'][:10],
                'target_positions': attentions['aggregated_attention']['target_token_info']
            }
        
        return result
    
    def _process_attention_scores(self, attentions, encoding, target_word):
        """
        Process the attention scores to make them more readable
        
        Args:
            attentions: Attention scores of each layer
            encoding: Encoded input
            target_word: target word
        
        Returns:
            The processed attention information
        """
        if attentions is None:
            return None
        
        input_ids = encoding['input_ids'][0]
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        target_token_positions = []
        for i, token in enumerate(tokens):
            if target_word.lower() in token.lower():
                target_token_positions.append(i)
        
        attention_info = {
            'tokens': tokens,
            'target_token_positions': target_token_positions,
            'layers': [],
            'aggregated_attention': {}
        }
        
        seq_len = len(tokens)
        all_layers_attention_sum = np.zeros((seq_len, seq_len))
        layer_attention_sums = []
        
        for layer_idx, layer_attention in enumerate(attentions):
            layer_attention = layer_attention[0]
            num_heads = layer_attention.shape[0]
            
            layer_attention_sum = np.zeros((seq_len, seq_len))
            
            layer_info = {
                'layer_index': layer_idx,
                'num_heads': num_heads,
                'attention_weights': {},
                'aggregated_target_attention': None
            }
            
            layer_target_attention = np.zeros(seq_len)
            
            for head_idx in range(num_heads):
                head_attention = layer_attention[head_idx].cpu().numpy()
                
                layer_attention_sum += head_attention
                
                target_attention = []
                for target_pos in target_token_positions:
                    target_to_others = head_attention[target_pos]
                    
                    layer_target_attention += target_to_others
                    
                    target_attention.append({
                        'target_position': target_pos,
                        'target_to_others': target_to_others.tolist(),
                        'others_to_target': head_attention[:, target_pos].tolist()
                    })
                
                layer_info['attention_weights'][f'head_{head_idx}'] = {
                    'target_attention': target_attention,
                    'attention_matrix': head_attention.tolist()
                }
            
            layer_info['aggregated_target_attention'] = layer_target_attention.tolist()
            attention_info['layers'].append(layer_info)
            
            all_layers_attention_sum += layer_attention_sum
            layer_attention_sums.append(layer_attention_sum)
        
        all_layers_target_attention = np.zeros(seq_len)
        for layer_info in attention_info['layers']:
            layer_target_attn = np.array(layer_info['aggregated_target_attention'])
            all_layers_target_attention += layer_target_attn
        
        if all_layers_target_attention.sum() > 0:
            all_layers_target_attention_norm = all_layers_target_attention / all_layers_target_attention.sum()
        else:
            all_layers_target_attention_norm = all_layers_target_attention
        
        token_attention_pairs = []
        for i, token in enumerate(tokens):
            attention_score = all_layers_target_attention[i]
            token_attention_pairs.append({
                'token': token,
                'position': i,
                'attention_score': float(attention_score),
                'normalized_score': float(all_layers_target_attention_norm[i]) if all_layers_target_attention.sum() > 0 else 0.0
            })
        
        token_attention_pairs.sort(key=lambda x: x['attention_score'], reverse=True)
        
        layer_normalized_attentions = []
        for layer_idx, layer_info in enumerate(attention_info['layers']):
            layer_target_attn = np.array(layer_info['aggregated_target_attention'])
            if layer_target_attn.sum() > 0:
                norm_attn = layer_target_attn / layer_target_attn.sum()
            else:
                norm_attn = layer_target_attn
            
            layer_normalized_attentions.append({
                'layer_index': layer_idx,
                'normalized_attention': norm_attn.tolist(),
                'top_tokens': []
            })
            
            layer_top_indices = np.argsort(layer_target_attn)[-10:][::-1]  # 取前10个
            for idx in layer_top_indices:
                if layer_target_attn[idx] > 0:
                    layer_normalized_attentions[-1]['top_tokens'].append({
                        'token': tokens[idx],
                        'position': idx,
                        'attention_score': float(layer_target_attn[idx])
                    })
        
        attention_info['aggregated_attention'] = {
           
            'all_layers_target_attention': all_layers_target_attention.tolist(),
            
            'all_layers_target_attention_normalized': all_layers_target_attention_norm.tolist(),
            
            'top_attended_tokens': token_attention_pairs[:20], 
            
            'layer_normalized_attentions': layer_normalized_attentions,
            
            'target_token_info': [
                {
                    'position': pos,
                    'token': tokens[pos],
                    'is_special_token': tokens[pos] in ['[CLS]', '[SEP]', '[PAD]', '[UNK]']
                }
                for pos in target_token_positions
            ]
        }
        
        return attention_info

    def get_aggregated_attention_analysis(self, context, target_word, top_k=10):

        result = self.predict_with_attention(context, target_word)
        
        if 'attention_scores' not in result or 'aggregated_attention' not in result['attention_scores']:
            return {
                'error': 'Unable to obtain aggregated attention information',
                'prediction': result
            }
        
        attention_info = result['attention_scores']['aggregated_attention']
        
        analysis = {
            'context': context,
            'target_word': target_word,
            'target_positions': attention_info['target_token_info'],
            
            'aggregated_scores': {
                'raw_scores': attention_info['all_layers_target_attention'],
                'normalized_scores': attention_info['all_layers_target_attention_normalized']
            },
            
            'most_attended_tokens': attention_info['top_attended_tokens'],
            
            'layer_attention_distribution': []
        }
        
        # 分析每层的注意力分布
        for layer_attn in attention_info['layer_normalized_attentions']:
            layer_idx = layer_attn['layer_index']
            
            target_self_attention = 0.0
            for pos_info in attention_info['target_token_info']:
                pos = pos_info['position']
                if pos < len(layer_attn['normalized_attention']):
                    target_self_attention += layer_attn['normalized_attention'][pos]
            
            analysis['layer_attention_distribution'].append({
                'layer': layer_idx,
                'self_attention_rate': float(target_self_attention),
                'top_tokens': layer_attn['top_tokens'][top_k]
            })
        
        return analysis

    def predict(self, context, target_word):

        if "roberta" in str(self.model.config.model_type).lower():
            marked_context = context.replace(target_word, f" [TGT]{target_word}[/TGT]")
        else:
            marked_context = context.replace(target_word, f"[TGT]{target_word}[/TGT]")
        
        encoding = self.tokenizer(
            marked_context,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        ).to(self.device)
        
        # pred
        with torch.no_grad():
            logits = self.model(**encoding).logits
            pred_id = torch.argmax(logits).item()
            pred_label = self.label_encoder.inverse_transform([pred_id])[0]
        
        target, definition = pred_label.split("::", 1)
        confidence = torch.softmax(logits, dim=1).max().item()

        all_probs = {}
        probabilities = torch.softmax(logits, dim=1)[0]
        for i, class_name in enumerate(self.label_encoder.classes_):
            all_probs[class_name] = probabilities[i].item()
            
        return {
            'target_word': target,
            'predicted_label': pred_label,
            'confidence': confidence,
            'context': context,
            'all_probabilities': all_probs
        }
    def eval(self, test_data_path, train_data_path=None):

        test_data = DataProcessor.load_json_data(test_data_path)
        tokenizer = self.tokenizer
        label_encoder = self.label_encoder

        is_roberta = "roberta" in str(self.model.config.model_type).lower()

        correct_by_class = defaultdict(int)
        total_by_class = defaultdict(int)
        tp_by_class = defaultdict(int)  
        fp_by_class = defaultdict(int)  
        fn_by_class = defaultdict(int)  
        
        accuracy_count = 0
        total_count = 0
        filtered_test_data = test_data
        
        # Filter the test data
        if train_data_path:
            train_data = DataProcessor.load_json_data(train_data_path)

            train_keys = set()
            for item in train_data:
                for key in item["correct_definition_key"]:
                    train_keys.add(key)

            # with open('bert-base-uncased_original_0.75semcor_second_unified_dropped_classes.json', 'r') as f:
            #     retained_classes = json.load(f)
            with open('bert_same_accuracy_other.json', 'r') as f:
                retained_classes = json.load(f)
            filtered_test_data = []
            for item in test_data:
                # if any(key in train_keys for key in item["correct_definition_key"]):
                #     filtered_test_data.append(item)
                glosses = item['correct_definitions_in_context']
                test_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
                # if any(label in retained_classes for label in test_labels) and any(key in train_keys for key in item["correct_definition_key"]):
                # if all(label not in retained_classes for label in glosses) and any(key in train_keys for key in item["correct_definition_key"]):
                if any(key in train_keys for key in item["correct_definition_key"]):
                    filtered_test_data.append(item)
                    
        total_by_class_in_train = defaultdict(int)
        for item in train_data:
            train_labels = [item['polysemous']['name'] + "::" + defn for defn in item['correct_definitions_in_context']]
            for label in train_labels:
                total_by_class_in_train[label] += 1
                
        candidate_senses_number = defaultdict(int)
        
        for item in tqdm(filtered_test_data, desc="评估中"):
            if is_roberta:
                marked_context = item["context"].replace(item["target"], f" [TGT]{item['target']}[/TGT]")
            else:
                marked_context = item["context"].replace(item["target"], f"[TGT]{item['target']}[/TGT]")
            inputs = tokenizer(
                marked_context,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ).to(self.device)
            
            with torch.no_grad():
                logits = self.model(**inputs).logits
                pred_id = torch.argmax(logits).item()
            
            predicted_label = label_encoder.inverse_transform([pred_id])[0]
            pred_def = predicted_label.split("::")[1]
            true_labels = [f"{item['polysemous']['name']}::{d}" for d in item["correct_definitions_in_context"]]
            true_defs = item["correct_definitions_in_context"]
            
            for l in true_labels:
                total_by_class[l] += 1
                candidate_senses_number[l] = len(item['polysemous']['sense_definitions_list'])
            
            predicted_class = predicted_label
            is_correct = pred_def in true_defs
            
            for true_label in true_labels:
                if true_label == predicted_class and is_correct:
                    tp_by_class[true_label] += 1
                else:
                    if true_label == predicted_class and not is_correct:
                        fp_by_class[predicted_class] += 1
                    fn_by_class[true_label] += 1

            if predicted_class not in true_labels and not is_correct:
                fp_by_class[predicted_class] += 1
            
            if is_correct:
                for true_label in true_labels:
                    correct_by_class[true_label] += 1
                accuracy_count += 1
            total_count += 1
        
        def safe_divide(a, b):
            return a / b if b > 0 else 0.0
        
        class_metrics = {}
        all_classes = set(list(total_by_class.keys()) + list(tp_by_class.keys()))
        
        f1_scores = []
        
        for label in all_classes:
            label_str = str(label)
            tp = tp_by_class.get(label_str, 0)
            fp = fp_by_class.get(label_str, 0)
            fn = fn_by_class.get(label_str, 0)
            
            precision = safe_divide(tp, tp + fp)
            recall = safe_divide(tp, tp + fn)
            f1 = safe_divide(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
            
            f1_scores.append(f1)
            
            acc = safe_divide(correct_by_class.get(label_str, 0), total_by_class.get(label_str, 1))
            class_metrics[label_str] = {
                "accuracy": acc,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support_in_all": total_by_class.get(label_str, 0),
                "candidate_senses_number": candidate_senses_number.get(label_str, 0),
                "support_in_train": total_by_class_in_train.get(label_str, 0),
                "tp": tp,
                "fp": fp,
                "fn": fn
            }
        
        total_accuracy = accuracy_count / total_count if total_count > 0 else 0.0
        
        macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            "class_wise_accuracy": class_metrics,
            "overall_accuracy": total_accuracy,
            "macro_f1": macro_f1
        }

class OptimizedModuleSelector:
    
    def __init__(self, model, monitor_layers=None):
        self.model = model
        self.monitor_layers = monitor_layers

    def select_target_modules(self):
        if self.monitor_layers is None:
            return self._select_default_critical_layers()
        
        target_modules = []
        all_modules = list(self.model.named_modules())
        
        for monitor_pattern in self.monitor_layers:
            matched = self._find_exact_matches(all_modules, monitor_pattern)
            target_modules.extend(matched)
        
        target_modules = list(dict(target_modules).items())
        
        return target_modules
    
    def _find_exact_matches(self, all_modules, pattern):
        matches = []
        
        for name, module in all_modules:
            if self._is_exact_leaf_match(name, pattern):
                matches.append((name, module))
        
        return matches
    
    def _is_exact_leaf_match(self, module_name, pattern):
        
        return (module_name.endswith(pattern) or 
                # f".{pattern}." in module_name or
                module_name == pattern)
    
    def _is_parent_module(self, module_name, pattern):

        parts = module_name.split('.')
        pattern_parts = pattern.split('.')
        
        if module_name == pattern:
            return True
        
        if len(parts) == len(pattern_parts) + 1: 
            parent_name = '.'.join(parts[:-1])
            return parent_name == pattern
        
        return False
    
    def _select_default_critical_layers(self):
        critical_patterns = {
            'classifier': 'classifier',
            'pooler': 'pooler', 
            'embeddings': 'embeddings',
            'encoder.layer.0': 'first Transformer layer',
            'encoder.layer.11': 'last Transformer layer', 
        }
        
        target_modules = []
        all_modules = list(self.model.named_modules())
        
        for pattern, description in critical_patterns.items():
            matches = self._find_exact_matches(all_modules, pattern)
            if matches:
                target_modules.extend(matches)
                print(f"  - {description}: {[name for name, _ in matches]}")
        
        return target_modules

class HookBasedMasker:
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.masking_strategies = {}
    
    def add_layer_mask(self, layer_pattern: List, mask_strategy="zero", mask_strength=1.0):

        module_selector = OptimizedModuleSelector(self.model, layer_pattern)
        for name, module in module_selector.select_target_modules():
            
            hook = self._create_masking_hook(name, mask_strategy, mask_strength)
            handle = module.register_forward_hook(hook)
            self.hooks.append(handle)
            self.masking_strategies[name] = (mask_strategy, mask_strength)
            print(f"为层 {name} 添加{mask_strategy} mask")
    
    def _create_masking_hook(self, layer_name, strategy, strength):
        def masking_hook(module, input, output):
            
            if isinstance(input, torch.Tensor):
               
                original_input = input
                masked_output = self._apply_mask_to_tensor(original_input, strategy, strength)
                return masked_output
                
            elif isinstance(input, tuple):
                original_inputs = input
                masked_outputs = []
                
                for i, out in enumerate(original_inputs):
                    if isinstance(out, torch.Tensor):
                        masked_out = self._apply_mask_to_tensor(out, strategy, strength)
                        masked_outputs.append(masked_out)
                    else:
                        masked_outputs.append(out)
            
                return tuple(masked_outputs)
                
            else:
                return output
        
        return masking_hook
    
    def _apply_mask_to_tensor(self, tensor, strategy, strength):
        """对单个Tensor应用mask"""
        if strategy == "zero":
            masked_tensor = torch.zeros_like(tensor)
            return tensor * (1 - strength) + masked_tensor * strength
            
        elif strategy == "reduce":
            return tensor * (1 - strength)
            
        elif strategy == "shuffle":
            if tensor.dim() >= 2:
                batch_size, hidden_size = tensor.shape[0], tensor.shape[-1]
                original_shape = tensor.shape
                flattened = tensor.reshape(batch_size, -1, hidden_size)
                indices = torch.randperm(flattened.size(1))
                shuffled = flattened[:, indices, :].reshape(original_shape)
                return tensor * (1 - strength) + shuffled * strength
            else:
                return tensor
                
        elif strategy == "freeze":
            return tensor + tensor * strength
            
        elif strategy == "noise":
            noise = torch.randn_like(tensor) * strength
            return tensor + noise
            
        elif strategy == "identity":
            return tensor
        else:
            return tensor

    def remove_all_masks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.masking_strategies.clear()
        print("All masks have been removed")

LAYER_CONFIGS = [
    {'layer_pattern': 'encoder.layer.7', 'strategy': 'identity', 'strength': 1.0},
    {'layer_pattern': 'encoder.layer.10', 'strategy': 'identity', 'strength': 1.0},
    {'layer_pattern': 'encoder.layer.11', 'strategy': 'identity', 'strength': 1.0},
    # {'layer_pattern': 'encoder.layer.8', 'strategy': 'identity', 'strength': 1.0},
]
class MaskableWSDPredictor(WSDPredictor):
    
    def __init__(self, model_path="./saved_models/best_model"):
        super().__init__(model_path)
        self.masker = HookBasedMasker(self.model)
        self.active_masks = {}
    
    def set_layer_mask(self, layer_configs=LAYER_CONFIGS):

        self.masker.remove_all_masks()
        
        for config in layer_configs:
            self.masker.add_layer_mask(
                [config['layer_pattern']],
                config.get('strategy', 'zero'),
                config.get('strength', 1.0)
            )
            self.active_masks[config['layer_pattern']] = config