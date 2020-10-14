from flask import Flask, request
import requests
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.tokenization import BertTokenizer

from run_classifier import convert_examples_to_features, InputExample, SelfDisclosureProcessor
import emot
import string
import re
import numpy as np
from nltk.tokenize import sent_tokenize
app = Flask(__name__)


model = BertForSequenceClassification.from_pretrained("output/sd-11", num_labels=3)

# n_gpu = torch.cuda.device_count()
device = torch.device("cuda")
model.to(device)


@app.route('/', methods=["POST"])
def hello_world():
    data = request.get_json()
    last_response = data.get("last_response")
    current_user_response = data.get("current_user_response")

    if has_emoticon(current_user_response) or has_exclamation_mark(current_user_response) or has_interjection(current_user_response):
        return "emotional"

    sentences = sent_tokenize(current_user_response)

    results = classify(last_response, sentences)
    print("Predicted results: ", results)

    result_labels = [item["label"] for item in results]
    if "emotional" in result_labels:
        return "emotional"
    elif "cognitive" in result_labels:
        return "cognitive"
    else:
        return "factual"


def classify(last_response, sentences):
    max_seq_length = 256
    binary_predict = False
    eval_batch_size = 16


    examples = convert_examples(last_response, sentences)
    label_list = SelfDisclosureProcessor.get_labels()
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased", do_lower_case=False)

    eval_features = convert_examples_to_features(
        examples, label_list, max_seq_length, tokenizer, binary_predict, inference=True)

    all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

    eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

    iterator = iter(eval_dataloader)
    eval_dataloader = next(iterator)

    input_ids, input_mask, segment_ids, label_ids = eval_dataloader

    input_ids = input_ids.to(device)
    segment_ids = segment_ids.to(device)
    label_ids = label_ids.to(device)
    input_mask = input_mask.to(device)

    logits = model(input_ids, segment_ids, input_mask, binary_pred=False)

    results = single_prediction(label_ids, logits)
    # labels = [label_list[item] for item in top_id_data]
    return results


def convert_examples(last_response, sentences):
    guid = "dev-1"
    label_1 = "INFERENCE"

    examples = []
    for i, sentence in enumerate(sentences):
        if i == 0:
            prev_text = "EMPTY"
        else:
            prev_text = " ".join(sentences[:i])

        context_text = last_response + " : " + prev_text
        current_text = sentences[i]
        examples.append(InputExample(guid=guid, text_a=context_text, text_b=current_text, label=label_1))

    return examples

def single_prediction(label_ids, logits):
    label_list = SelfDisclosureProcessor.get_labels()

    # iterator = iter(zip(label_ids, logits))
    # tgt_label, pred_da = next(iterator)

    results = []
    for tgt_label, pred_da in zip(label_ids, logits):
        top_k_value, top_k_ind = torch.topk(pred_da, 1)
        k_value = torch.sigmoid(top_k_value).view(-1).data.cpu().numpy()[0]
        top_id_data = top_k_ind.view(-1).data.cpu().numpy()[0]

        result = dict()
        result["label"] = label_list[top_id_data]
        result["confidence"] = k_value

        results.append(result)
    return results


def binary_prediction(label_ids, logits):
    iterator = iter(zip(label_ids, logits))
    tgt_label, pred_da = next(iterator)

    result = dict()
    label_list = SelfDisclosureProcessor.get_labels()

    tgt_ids = []
    for i in np.nonzero(tgt_label).view(-1).data.cpu().numpy():
        tgt_ids.append(i)

    top_k_value, top_k_ind = torch.topk(pred_da, 2)
    top_id_data = []
    for k_value, k_ind in zip(torch.sigmoid(top_k_value).view(-1).data.cpu().numpy(),
                              top_k_ind.view(-1).data.cpu().numpy()):
        if k_value > 0.5:
            result[label_list[k_ind]] = k_value
            # top_id_data.append(k_ind)

    if len(top_id_data) == 0:
        # top_id_data.append(top_k_ind.view(-1).data.cpu().numpy()[0])
        k_ind = top_k_ind.view(-1).data.cpu().numpy()[0]
        k_value = torch.sigmoid(top_k_value).view(-1).data.cpu().numpy()[0]

        result[label_list[k_ind]] = k_value

    return result


def has_emoticon(text):
    print(emot.__version__)

    result = emot.emoticons(text)
    print("has_emoticon: ", result)
    if isinstance(result, list):
        return True in [item['flag'] for item in result]
    elif isinstance(result, dict):
        return result['flag']
    else:
        return False


def has_exclamation_mark(text):
    return "!" in text


def has_interjection(text):
    return re.search("|".join([
        r"\b(ha( )?){1,}\b",
        r"\b(wow)\b",
        r"\b(lol)\b"
    ]), text, flags=re.IGNORECASE)


def analysis():
    from sklearn.metrics import confusion_matrix, f1_score
    all_predicted = [
        1, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 2, 2, 2, 0, 2, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 2, 1, 1, 2, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 2, 2, 0, 1, 0, 0, 2, 1, 1, 2, 1, 2, 1, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 1, 0, 0, 0, 0, 1, 1, 0, 0]
    all_actual = [
        1, 1, 0, 1, 1, 1, 0, 2, 0, 0, 1, 0, 0, 0, 0, 1, 0, 2, 0, 1, 2, 2, 2, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 2, 0, 2, 1, 0, 2, 0, 2, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 2, 1, 0, 1, 1, 2, 0, 0, 0, 1, 0, 0, 1, 2, 2, 0, 0, 0, 0, 2, 1, 1, 2, 1, 2, 2, 2, 0, 2, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0]
    confusion_matrix(all_actual, all_predicted)
    f1_weighted = f1_score(all_actual, all_predicted, 'weighted')
    print(f1_weighted)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=2345)
    # hello_world()
    # analysis()