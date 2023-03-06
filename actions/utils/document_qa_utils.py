"""
Name : deployment.py
Author  : 北在哪
Contect : 1439187192@qq.com
Time    : 2023/2/27 11:20
Desc:
"""
# from model import BertForQuestionAnswering
import torch
from transformers import BertTokenizer, BasicTokenizer, BertForQuestionAnswering
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import timeit
import collections
import math
from pydantic import BaseModel


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs


def _is_whitespace(c):
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def get_final_text(pred_text, orig_text, do_lower_case):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        # print(f"Unable to find text: {pred_text} in {orig_text}")
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        # print(f"Length not equal after stripping spaces: {orig_ns_text} vs {tok_ns_text}")
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        print("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        print("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position: (orig_end_position + 1)]
    return output_text


class SquadExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        document_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
    """

    def __init__(
            self,
            document_id,
            question_text,
            context_text,
    ):
        self.document_id = document_id
        self.question_text = question_text
        self.context_text = context_text

        doc_tokens = []
        char_to_word_offset = []
        prev_is_whitespace = True

        # Split on whitespace so that different tokens may be attributed to their original position.
        # 在空白处拆分，以便不同的标记可以归属于其原始位置。
        for c in self.context_text:
            if _is_whitespace(c):
                prev_is_whitespace = True
            else:
                if prev_is_whitespace:
                    doc_tokens.append(c)
                else:
                    doc_tokens[-1] += c
                prev_is_whitespace = False
            char_to_word_offset.append(len(doc_tokens) - 1)

        self.doc_tokens = doc_tokens
        self.char_to_word_offset = char_to_word_offset


class SquadFeaturesOrig(object):
    """
    Single squad example features to be fed to a model.
    Those features are model-specific and can be crafted from :class:`~transformers.data.processors.squad.SquadExample`
    using the :method:`~transformers.data.processors.squad.squad_convert_examples_to_features` method.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
        token_type_ids: Segment token indices to indicate first and second portions of the inputs.
        cls_index: the index of the CLS token.
        example_index: the index of the example
        unique_id: The unique Feature identifier
        paragraph_len: The length of the context
        tokens: list of tokens corresponding to the input ids
        token_to_orig_map: mapping between the tokens and the original text, needed in order to identify the answer.
    """

    def __init__(
            self,
            input_ids,
            attention_mask,
            token_type_ids,
            cls_index,
            example_index,
            unique_id,
            paragraph_len,
            tokens,
            token_to_orig_map
    ):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.cls_index = cls_index

        self.example_index = example_index
        self.unique_id = unique_id
        self.paragraph_len = paragraph_len
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map


class SquadResult(object):
    """
    Constructs a SquadResult which can be used to evaluate a model's output on the SQuAD dataset.

    Args:
        unique_id: The unique identifier corresponding to that example.
        start_logits: The logits corresponding to the start of the answer
        end_logits: The logits corresponding to the end of the answer
    """

    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logits=None):
        self.unique_id = unique_id
        self.start_logits = start_logits
        self.end_logits = end_logits

        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logits


def create_examples(input_datas):
    """
    input_data = [{'title': '',
                  'document': '',
                  'document_id': '',
                  'question': ''}, ...]
    """
    examples = []
    if isinstance(input_datas[0], BaseModel):
        for input_data in input_datas:
            title = input_data.title
            if title != '':
                context_text = title + input_data.document
            else:
                context_text = input_data.document

            question_text = input_data.question
            document_id = input_data.document_id

            example = SquadExample(
                document_id=document_id,
                question_text=question_text,
                context_text=context_text,
            )
            examples.append(example)
    else:
        for input_data in input_datas:
            title = input_data["title"] if 'title' in input_data else ''
            if title != '':
                context_text = title + input_data["document"]
            else:
                context_text = input_data["document"]

            question_text = input_data["question"]
            document_id = input_data["document_id"]

            example = SquadExample(
                document_id=document_id,
                question_text=question_text,
                context_text=context_text,
            )
            examples.append(example)

    return examples


def squad_convert_example_to_features_init(tokenizer_for_convert):
    global tokenizer
    tokenizer = tokenizer_for_convert


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer, orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start: (new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def squad_convert_example_to_features_orig(example, max_seq_length, max_query_length):
    features = []

    tok_to_orig_index = []  # tokenize后的字符对应doc_tokens中的哪一片段
    orig_to_tok_index = []  # doc_tokens中的各个片段在all_doc_tokens的起始位置索引
    all_doc_tokens = []  # tokenize后的所有字符
    for (i, token) in enumerate(example.doc_tokens):
        orig_to_tok_index.append(len(all_doc_tokens))
        sub_tokens = tokenizer.tokenize(token)
        for sub_token in sub_tokens:
            tok_to_orig_index.append(i)
            all_doc_tokens.append(sub_token)

    truncated_query = tokenizer.encode(example.question_text, add_special_tokens=False, max_length=max_query_length)
    sequence_added_tokens = (
        tokenizer.max_len - tokenizer.max_len_single_sentence + 1
        if "roberta" in str(type(tokenizer)) or "camembert" in str(type(tokenizer))
        else tokenizer.max_len - tokenizer.max_len_single_sentence
    )
    sequence_pair_added_tokens = tokenizer.max_len - tokenizer.max_len_sentences_pair

    encoded_dict = {}
    try:
        encoded_dict = tokenizer.encode_plus(
            truncated_query if tokenizer.padding_side == "right" else all_doc_tokens,
            all_doc_tokens if tokenizer.padding_side == "right" else truncated_query,
            max_length=max_seq_length,
            pad_to_max_length=True,
            truncation_strategy="only_second" if tokenizer.padding_side == "right" else "only_first",
            return_token_type_ids=True,
        )
    except Exception as e:
        print(f'处理{example.document_id}时出现{e}!')

    paragraph_len = min(len(all_doc_tokens),
                        max_seq_length - len(truncated_query) - sequence_pair_added_tokens)

    if tokenizer.pad_token_id in encoded_dict["input_ids"]:
        if tokenizer.padding_side == "right":
            non_padded_ids = encoded_dict["input_ids"][: encoded_dict["input_ids"].index(tokenizer.pad_token_id)]
        else:
            last_padding_id_position = (
                    len(encoded_dict["input_ids"]) - 1 - encoded_dict["input_ids"][::-1].index(
                tokenizer.pad_token_id)
            )
            non_padded_ids = encoded_dict["input_ids"][last_padding_id_position + 1:]

    else:
        non_padded_ids = encoded_dict["input_ids"]

    tokens = tokenizer.convert_ids_to_tokens(non_padded_ids)

    token_to_orig_map = {}
    for i in range(paragraph_len):
        index = len(truncated_query) + sequence_added_tokens + i if tokenizer.padding_side == "right" else i
        token_to_orig_map[index] = tok_to_orig_index[i]

    encoded_dict["paragraph_len"] = paragraph_len
    encoded_dict["tokens"] = tokens
    encoded_dict["token_to_orig_map"] = token_to_orig_map
    encoded_dict["length"] = paragraph_len

    # Identify the position of the CLS token
    cls_index = encoded_dict["input_ids"].index(tokenizer.cls_token_id)

    features.append(
        SquadFeaturesOrig(
            encoded_dict["input_ids"],
            encoded_dict["attention_mask"],
            encoded_dict["token_type_ids"],
            cls_index,
            example_index=0,
            # Can not set unique_id and example_index here. They will be set after multiple processing.
            unique_id=0,
            paragraph_len=encoded_dict["paragraph_len"],
            tokens=encoded_dict["tokens"],
            token_to_orig_map=encoded_dict["token_to_orig_map"],
        )
    )
    return features


def squad_convert_examples_to_features_orig(examples, tokenizer, max_seq_length, max_query_length):
    """
    Converts a list of examples into a list of features that can be directly given as input to a model.
    It is model-dependant and takes advantage of many of the tokenizer's features to create the model's inputs.

    Args:
        examples: list of :class:`~transformers.data.processors.squad.SquadExample`
        tokenizer: an instance of a child of :class:`~transformers.PreTrainedTokenizer`
        max_seq_length: The maximum sequence length of the inputs.
        max_query_length: The maximum length of the query.


    Returns:
        list of :class:`~transformers.data.processors.squad.SquadFeatures`

    Example::

        processor = SquadV2Processor()
        examples = processor.get_dev_examples(data_dir)

        features = squad_convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
        )
    """

    features = []
    # 将tokenizer变量全局化
    squad_convert_example_to_features_init(tokenizer)
    for example in examples:
        example_features = squad_convert_example_to_features_orig(example, max_seq_length, max_query_length)
        features.append(example_features)

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_cls_index = torch.tensor([f.cls_index for f in features], dtype=torch.long)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(
        all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, all_cls_index
    )

    return features, dataset


def squad_convert_examples_to_features_orig_onnx(examples, tokenizer, max_seq_length, max_query_length):
    """
    onnx 推理用
    """

    features = []
    # 将tokenizer变量全局化
    squad_convert_example_to_features_init(tokenizer)
    for example in examples:
        example_features = squad_convert_example_to_features_orig(example, max_seq_length, max_query_length)
        features.append(example_features)

    new_features = []
    unique_id = 1000000000
    example_index = 0
    for example_features in features:
        if not example_features:
            continue
        for example_feature in example_features:
            example_feature.example_index = example_index
            example_feature.unique_id = unique_id
            new_features.append(example_feature)
            unique_id += 1
        example_index += 1
    features = new_features
    del new_features

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_masks = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

    return features, all_input_ids, all_attention_masks, all_token_type_ids, all_example_index


def load_examples(tokenizer, input_datas, max_seq_length=512, max_query_length=64):
    examples = create_examples(input_datas)

    features, dataset = squad_convert_examples_to_features_orig(
        examples=examples,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        max_query_length=max_query_length
    )

    return dataset, examples, features


def load_examples_onnx(tokenizer, input_datas, max_seq_length=512, max_query_length=64):
    examples = create_examples(input_datas)

    features, all_input_ids, all_attention_masks, all_token_type_ids, all_example_index = \
        squad_convert_examples_to_features_orig_onnx(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            max_query_length=max_query_length
        )

    return all_input_ids, all_attention_masks, all_token_type_ids, all_example_index, examples, features


def compute_predictions_logits(
        all_examples,
        all_features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer,
):
    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction", ["feature_index", "start_index", "end_index", "start_logit", "end_logit"]
    )

    all_predictions = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]
        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index],
                        )
                    )
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit,
                )
            )
        prelim_predictions = sorted(prelim_predictions, key=lambda x: (x.start_logit + x.end_logit), reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"]
        )

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index: (pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start: (orig_doc_end + 1)]

                tok_text = tokenizer.convert_tokens_to_string(tok_tokens)

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(_NbestPrediction(text=final_text, start_logit=pred.start_logit, end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(_NbestPrediction(text="no answer", start_logit=null_start_logit, end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest) == 1:
                nbest.insert(0, _NbestPrediction(text="no answer", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(_NbestPrediction(text="no answer", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.document_id] = {'text': nbest_json[0]["text"],
                                                    'score': nbest_json[0]["probability"]}
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - best_non_null_entry.end_logit
            if score_diff > null_score_diff_threshold:
                all_predictions[example.document_id] = {'text': "no answer",
                                                        'score': 0}
            else:
                all_predictions[example.document_id] = {'text': best_non_null_entry.text,
                                                        'score': nbest_json[0]["probability"]}

    return all_predictions


def predict(model, tokenizer, input_data, n_best_size=10, max_answer_length=512, do_lower_case=True,
            version_2_with_negative=True, null_score_diff_threshold=0):
    dataset, examples, features = load_examples(tokenizer, input_data)

    eval_batch_size = 10

    eval_dataloader = DataLoader(dataset, batch_size=eval_batch_size)

    all_results = []
    start_time = timeit.default_timer()

    for batch in eval_dataloader:
        model.eval()
        batch = tuple(t.cuda() for t in batch)

        with torch.no_grad():
            inputs = {
                "input_ids": batch[0],
                "attention_mask": batch[1],
                "token_type_ids": batch[2],
            }

            example_indices = batch[3]

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)

            output = [to_list(output[i]) for output in outputs]

            start_logits, end_logits = output
            result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    predictions = compute_predictions_logits(
        examples,
        features,
        all_results,
        n_best_size,
        max_answer_length,
        do_lower_case,
        version_2_with_negative,
        null_score_diff_threshold,
        tokenizer
    )

    evalTime = timeit.default_timer() - start_time
    print(f"  Evaluation done in total {evalTime:.3f} secs ({evalTime / len(dataset):.3f} sec per example)")

    return predictions


if __name__ == '__main__':
    model_name_or_path = '/code/MODEL/luhua-chinese_pretrain_mrc_macbert_large'
    model = BertForQuestionAnswering.from_pretrained(model_name_or_path)
    model.cuda()
    tokenizer = BertTokenizer.from_pretrained(model_name_or_path, do_lower_case=True)
    input_data = [{'title': '',
                   'document': '关于丽湖校区北区楼宇11月12日凌晨时段停水的通知各位师生：因四方楼水泵房水箱进水改管需要，丽湖校区部分楼宇供水将受到影响，具体影响时间及影响范围如下：影响时间：11月12日（星期六）00：00-11月12日（星期六）7:00影响范围：梧桐树、青冈栎、三角梅、冬青树、紫罗兰、伐木餐厅、伐檀餐厅、留学生活动中心、公共教学楼(四方楼）、明理楼、明德楼、明律楼、启明楼（中央图书馆）、守正楼。请各涉及楼栋师生注意做好停水准备，给您带来不便，敬请谅解！如有疑问，请联系中航物业（丽湖校区）24小时客服中心值班电话：0755-21672017。丽湖校区管理办公室2022年11月10日关于丽湖校区北区楼宇11月12日凌晨时段停水的通知各位师生：因四方楼水泵房水箱进水改管需要，丽湖校区部分楼宇供水将受到影响，具体影响时间及影响范围如下：影响时间：11月12日（星期六）00：00-11月12日（星期六）7:00影响范围：梧桐树、青冈栎、三角梅、冬青树、紫罗兰、伐木餐厅、伐檀餐厅、留学生活动中心、公共教学楼(四方楼）、明理楼、明德楼、明律楼、启明楼（中央图书馆）、守正楼。请各涉及楼栋师生注意做好停水准备，给您带来不便，敬请谅解！如有疑问，请联系中航物业（丽湖校区）24小时客服中心值班电话：0755-21672017。丽湖校区管理办公室2022年11月10日',
                   'document_id': 'life_3_seg_0',
                   'question': '宿舍为什么停水？'},
                  ]
    res = predict(model, tokenizer, input_data)
    print(res)
