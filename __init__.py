from training import *
from predictor import *
import argparse
from tqdm import tqdm

def predictor_layer_analysis(predictor, test_data_path, train_data_path=None, layer_num=12):

    EACH_LAYER_CONFIGS = [ {'layer_pattern': f'encoder.layer.{i}', 'strategy': 'identity', 'strength': 1.0} for i in range(layer_num)]


    test_data = DataProcessor.load_json_data(test_data_path)
    filtered_test_data = test_data 

    if train_data_path:
        train_data = DataProcessor.load_json_data(train_data_path)
        train_keys = set()
        for item in train_data:
            for key in item["correct_definition_key"]:
                train_keys.add(key)
        filtered_test_data = []
        for item in test_data:
            if any(key in train_keys for key in item["correct_definition_key"]):
                filtered_test_data.append(item)
    baseline_positive_results = []
    baseline_negative_results = []
    for item in tqdm(filtered_test_data, desc="BaseLine"):
        result = predictor.predict(item['context'], item['target'])
        true_labels = [f"{item['polysemous']['name']}::{t}" for t in item["correct_definitions_in_context"]]
        predicted_label = result['predicted_label']
        if predicted_label in true_labels:
            baseline_positive_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'max_probabilities': max(result['all_probabilities'].values())
            })
        else:
            baseline_negative_results.append({
                'target': item['target'],
                'context': item['context'],
                'target_word': item['polysemous']['name'],
                'true_labels': true_labels,
                'predicted_label': predicted_label,
                'predicted_probabilities': result['all_probabilities'][predicted_label],
                'true_probabilities': max([result['all_probabilities'][true_label] for true_label in true_labels]),
                'max_probabilities': max(result['all_probabilities'].values())
            })
        del(result)
    del filtered_test_data

    positive_results_per_layer = []
    negative_results_per_layer = []
    for i in range(layer_num - 1, 0, -1):
        layer_config = EACH_LAYER_CONFIGS[i:layer_num]
        predictor.set_layer_mask(layer_config)
        positive_layer_results = []
        for item in tqdm(baseline_positive_results, desc="Positive Item"):
            result = predictor.predict(item['context'], item['target'])
            positive_layer_results.append({
                'target_word': item['target_word'],
                'predicted_label': item['predicted_label'],
                'predicted_probabilities': result['all_probabilities'][item['predicted_label']],
                'max_probabilities': max(result['all_probabilities'].values())
            })
            del result
        positive_results_per_layer.append(positive_layer_results)
        negative_layer_results = []
        for item in tqdm(baseline_negative_results, desc="Negative Item"):
            result = predictor.predict(item['context'], item['target'])
            negative_layer_results.append({
                'target_word': item['target_word'],
                'predicted_label': item['predicted_label'],
                'predicted_probabilities': result['all_probabilities'][item['predicted_label']],
                'true_probabilities': max([result['all_probabilities'][true_label] for true_label in item['true_labels']]),
                'max_probabilities': max(result['all_probabilities'].values())
            })
            del result
        negative_results_per_layer.append(negative_layer_results)
    positive_prob_per_layer = []
    negative_prob_per_layer = []
    for layer_results in positive_results_per_layer:
        layer_predicted_label_probs = [item['predicted_probabilities'] for item in layer_results]
        mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
        layer_max_probs = [item['max_probabilities'] for item in layer_results]
        mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
        positive_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 
                                        'mean_max_probabilities': mean_max_prob})
    positive_prob_per_layer.reverse()
    layer_predicted_label_probs = [item['predicted_probabilities'] for item in baseline_positive_results]
    mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
    layer_max_probs = [item['max_probabilities'] for item in baseline_positive_results]
    mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
    positive_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob,
                                    'mean_max_probabilities': mean_max_prob})
    for layer_results in negative_results_per_layer:
        layer_predicted_label_probs = [item['predicted_probabilities'] for item in layer_results]
        mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
        layer_true_label_probs = [item['true_probabilities'] for item in layer_results]
        mean_true_prob = sum(layer_true_label_probs) / len(layer_true_label_probs)
        layer_max_probs = [item['max_probabilities'] for item in layer_results]
        mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
        negative_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 'mean_true_prob': mean_true_prob,
                                        'mean_max_probabilities': mean_max_prob})
    negative_prob_per_layer.reverse()
    layer_predicted_label_probs = [item['predicted_probabilities'] for item in baseline_negative_results]
    mean__predicted_prob = sum(layer_predicted_label_probs) / len(layer_predicted_label_probs)
    layer_true_label_probs = [item['true_probabilities'] for item in baseline_negative_results]
    mean_true_prob = sum(layer_true_label_probs) / len(layer_true_label_probs)
    layer_max_probs = [item['max_probabilities'] for item in baseline_negative_results]
    mean_max_prob = sum(layer_max_probs) / len(layer_max_probs)
    negative_prob_per_layer.append({'mean_predicted_prob': mean__predicted_prob, 'mean_true_prob': mean_true_prob,
                                    'mean_max_probabilities': mean_max_prob})
    return {
        'positive_probs': positive_prob_per_layer,
        'negative_probs': negative_prob_per_layer
    }

def whr(predector, k = 10):
    with open('annotated/annotated_data.json', 'r', encoding='utf-8') as f:
        annotated_data = json.load(f)
    special_tokens = {'.':[], ',':[], '[pad]':[], '[cls]':[], '[sep]':[], '[unk]':[], '[tgt]':[], '[/tgt]':[]}
    ks0={'keyword1':[], 'keyword2':[], 'keyword3':[]}
    ks1={'keyword1':[], 'keyword2':[], 'keyword3':[]}
    wks0={'keyword1':0, 'keyword2':0, 'keyword3':0}
    wks1={'keyword1':0, 'keyword2':0, 'keyword3':0}
    N0 = 0
    N1 = 0

    for item in tqdm(annotated_data, desc="HR Evaluation"):
        acc = item['accuracy']
        if acc:
            N0 += 1
        else:
            N1 += 1
        keywords = item['keywords']
        analysis = predector.get_aggregated_attention_analysis(item['context'], item['target'])
        for i, token_info in enumerate(analysis['most_attended_tokens']):
            token = token_info['token']
            if token.lower() in special_tokens:
                special_tokens[token.lower()].append(i + 1)
                continue
            for j, keyword in enumerate(keywords):
                if token.lower() in keyword.lower():
                    if acc:
                        ks0[f'keyword{j+1}'].append(i + 1)
                        wks0[f'keyword{j+1}'] += 1
                    else:
                        ks1[f'keyword{j+1}'].append(i + 1)
                        wks1[f'keyword{j+1}'] += 1
    N = len(annotated_data)
    acc0_hr1 = sum([max(0, k - k1 + 1) for k1 in ks0['keyword1']]) / N0 / k
    acc0_hr2 = sum([max(0, k - k2 + 1) for k2 in ks0['keyword2']]) / N0 / k
    acc0_hr3 = sum([max(0, k - k3 + 1) for k3 in ks0['keyword3']]) / N0 / k
    acc1_hr1 = sum([max(0, k - k1 + 1) for k1 in ks1['keyword1']]) / N1 / k
    acc1_hr2 = sum([max(0, k - k2 + 1) for k2 in ks1['keyword2']]) / N1 / k
    acc1_hr3 = sum([max(0, k - k3 + 1) for k3 in ks1['keyword3']]) / N1 / k
    acc0_whr0 = wks0['keyword1'] / N0
    acc0_whr1 = wks0['keyword2'] / N0
    acc0_whr2 = wks0['keyword3'] / N0
    acc1_whr0 = wks1['keyword1'] / N1
    acc1_whr1 = wks1['keyword2'] / N1
    acc1_whr2 = wks1['keyword3'] / N1
    print(f"Weighted Accuracy 0: WHR@{k} for keyword1: {acc0_whr0}, keyword2: {acc0_whr1}, keyword3: {acc0_whr2}")
    print(f"Weighted Accuracy 1: WHR@{k} for keyword1: {acc1_whr0}, keyword2: {acc1_whr1}, keyword3: {acc1_whr2}")
    print(f"Accuracy 0: HR@{k} for keyword1: {acc0_hr1}, keyword2: {acc0_hr2}, keyword3: {acc0_hr3}")
    print(f"Accuracy 1: HR@{k} for keyword1: {acc1_hr1}, keyword2: {acc1_hr2}, keyword3: {acc1_hr3}")
    special_tokens_average_positions = {}
    for token, positions in special_tokens.items():
        if positions:
            average_position = sum(positions) / len(positions)
        else:
            average_position = -1
        special_tokens_average_positions[token] = average_position
    print(f"Special Tokens Attention Average Positions:", special_tokens_average_positions)
    pass

def eval(model_path, test_data_path, train_data_path, result_name='all_results'):
    predictor = WSDPredictor(model_path)
    metrics = predictor.eval(test_data_path, train_data_path)
    print(f"acc:{metrics['overall_accuracy']}, f1:{metrics['macro_f1']}")
    path = predictor.path + f'/{result_name}.json'
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=4)

def prob_per_layer(model_path, test_data_path, train_data_path, layer_num=12):
    predictor = MaskableWSDPredictor(model_path)
    results = predictor_layer_analysis(predictor, test_data_path, train_data_path, layer_num=layer_num)
    path = predictor.path + '/mean_prob_per_layer.json'
    results_dict = {'positive items': {f'layer.{i}': result for i, result in enumerate(results['positive_probs'])},
                    'negative items': {f'layer.{i}': result for i, result in enumerate(results['negative_probs'])}}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)

def train(train_data_path, training_artifacts_path=None, original_data_path='data\semcor.json'):
    config = Config()
    trainer = WSDTrainer(config)
    trainer.prepare_data(train_data_path
                         ,training_artifacts_path
                        , original_data_path=original_data_path
                         )
    trainer.train()

def compute_whr(model_path):
    predictor = MaskableWSDPredictor(model_path)
    whr(predictor, k=10)

def main():
    parser = argparse.ArgumentParser(
        description="WSD Experiment Pipeline Runner"
    )

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["eval", "prob", "train", "whr"],
        help="Task to run"
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name or path"
    )

    parser.add_argument(
        "--test_data",
        type=str,
        default=None,
        help="Path to test data"
    )

    parser.add_argument(
        "--train_data",
        type=str,
        default=None,
        help="Path to train data"
    )

    parser.add_argument(
        "--artifacts",
        type=str,
        default=None,
        help="Path to save training artifacts (optional)"
    )

    parser.add_argument(
        "--orig_data",
        type=str,
        default="SemCor",
        help="Path to original dataset (SemCor by default)"
    )

    args = parser.parse_args()

    # ===============================
    #             Tasks
    # ===============================

    if args.task == "eval":
        if not (args.model and args.test_data and args.train_data):
            raise ValueError("eval requires --model --test_data --train_data")
        eval(args.model, args.test_data, args.train_data)

    elif args.task == "prob":
        if not (args.model and args.test_data and args.train_data):
            raise ValueError("prob requires --model --test_data --train_data")
        prob_per_layer(args.model, args.test_data, args.train_data)

    elif args.task == "train":
        if not args.train_data:
            raise ValueError("train requires --train_data")
        train(args.train_data, args.artifacts, args.orig_data)

    elif args.task == "whr":
        if not args.model:
            raise ValueError("whr requires --model")
        compute_whr(args.model)


if __name__ == "__main__":
    main()