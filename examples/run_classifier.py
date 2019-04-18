# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import concurrent.futures
import copy
import csv
import logging
import os
import random
import sys
from functools import partial

import distiller
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.nn import CrossEntropyLoss
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from pytorch_pretrained_bert import BertForSequenceClassification
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import env_enabled, ENV_OPENAIGPT_GELU, ENV_DISABLE_APEX, BlendCNN, \
    BlendCNNForSequencePairClassification, DISTILLER_WEIGHTS_NAME, WEIGHTS_NAME, ACT2FN
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.tokenization import BertTokenizer

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class MrpcProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_dir, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            text_b = line[4]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class MnliProcessor(DataProcessor):
    """Processor for the MultiNLI data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
            "dev_matched")

    def get_labels(self):
        """See base class."""
        return ["contradiction", "entailment", "neutral"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, line[0])
            text_a = line[8]
            text_b = line[9]
            label = line[-1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


class ColaProcessor(DataProcessor):
    """Processor for the CoLA data set (GLUE version)."""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_labels(self):
        """See base class."""
        return ["0", "1"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[3]
            label = line[1]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples


class CustomProcessor(DataProcessor):
    """Processor for custom data set"""

    def __init__(self, labels):
        self.labels = labels

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return self.labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text_a = line[0]
            text_b = line[1]
            label = line[2]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples


def convert_example_to_feature(example, label_map, max_seq_length, tokenizer):
    try:
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = None
        if example.label is not None:
            label_id = label_map[example.label]

        return InputFeatures(input_ids=input_ids,
                             input_mask=input_mask,
                             segment_ids=segment_ids,
                             label_id=label_id)
    except Exception as err:
        print('Encountered error when converting {}: {}'.format(example, err), file=sys.stderr)
        zeros = [0] * max_seq_length
        return InputFeatures(input_ids=zeros,
                             input_mask=zeros,
                             segment_ids=zeros,
                             label_id=None)


def convert_examples_to_features(examples,
                                 label_list,
                                 max_seq_length,
                                 tokenizer,
                                 num_workers=0,
                                 desc=None,
                                 verbose=False):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}
    iterable = examples
    convert = partial(convert_example_to_feature,
                      label_map=label_map,
                      max_seq_length=max_seq_length,
                      tokenizer=tokenizer)

    if num_workers > 0:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            if verbose:
                iterable = tqdm(examples, dynamic_ncols=True, desc='Submit jobs')
            iterable = executor.map(convert,
                                    iterable,
                                    chunksize=max(512, len(examples) // num_workers // num_workers))
            if verbose:
                iterable = tqdm(iterable, dynamic_ncols=True, desc=desc, total=len(examples))
            return list(iterable)
    else:
        if verbose:
            iterable = tqdm(examples, dynamic_ncols=True, desc=desc)

        return list(map(convert, iterable))


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def accuracy(out, labels):
    outputs = np.argmax(out, axis=-1)
    return np.sum(outputs == labels)


def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return 1.0 - x


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        choices=[
                            "bert-base-uncased",
                            "bert-large-uncased",
                            "bert-base-cased",
                            "bert-large-cased",
                            "bert-base-multilingual-uncased",
                            "bert-base-multilingual-cased",
                            "bert-base-chinese",
                        ],
                        help="Bert pre-trained model selected in the list")
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True,
                        help="The name of the task to train.")
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--labels",
                        nargs='+',
                        default=['0', '1'],
                        help="labels")

    ## Other parameters
    parser.add_argument("--max_seq_length",
                        default=128,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--do_train",
                        action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval",
                        action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test",
                        action='store_true',
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_distill",
                        action='store_true',
                        help="Whether to run distillation.")
    parser.add_argument("--blendcnn_channels",
                        nargs='+',
                        default=(100,) * 8,
                        help="BlendCNN channels.")
    parser.add_argument("--blendcnn_act",
                        default='relu',
                        choices=list(ACT2FN.keys()),
                        help="BlendCNN activation function.")
    parser.add_argument('--blendcnn_dropout',
                        action='store_true',
                        help="Whether to use dropout in BlendCNN")
    parser.add_argument('--blendcnn_pair',
                        action='store_true',
                        help="Whether to use BlendCNNForSequencePairClassification")
    parser.add_argument("--export_onnx",
                        action='store_true',
                        help="Whether to export model to onnx format.")
    parser.add_argument("--onnx_framework",
                        choices=[
                            "caffe2",
                        ],
                        help="Select the ONNX framework to run eval")
    parser.add_argument("--eval_interval",
                        default=1000,
                        type=int,
                        help="Specify eval interval during training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size",
                        default=32,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size",
                        default=8,
                        type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs",
                        default=3.0,
                        type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion",
                        default=0.1,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    distiller.knowledge_distillation.add_distillation_args(parser)
    args = parser.parse_args()

    processors = {
        "cola": ColaProcessor,
        "mnli": MnliProcessor,
        "mrpc": MrpcProcessor,
        "custom": lambda: CustomProcessor(args.labels),
    }

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = int(args.train_batch_size / args.gradient_accumulation_steps)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not any((args.do_train, args.do_eval, args.do_test, args.do_distill, args.export_onnx)):
        raise ValueError("At least one of `do_train`, `do_eval`, `do_test`, `do_distill`, `export_onnx` must be True.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    os.makedirs(args.output_dir, exist_ok=True)

    task_name = args.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()
    label_list = processor.get_labels()
    num_labels = len(label_list)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    global_step = 0
    loss = 0
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
    onnx_model_file = os.path.join(args.output_dir, "model.onnx")
    eval_data = None

    if args.do_train:
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              cache_dir=PYTORCH_PRETRAINED_BERT_CACHE / 'distributed_{}'.format(
                                                                  args.local_rank),
                                                              num_labels=num_labels)
        model = convert_model(args, model, device, n_gpu)

        tensorboard_log_dir = os.path.join(args.output_dir, './log')
        os.makedirs(tensorboard_log_dir, exist_ok=True)
        tensorboard_logger = SummaryWriter(tensorboard_log_dir)

        if args.do_eval and do_eval_or_test(args) and eval_data is None:
            eval_data = prepare(args, processor, label_list, tokenizer, 'dev')

        global_step, loss = train(args,
                                  model,
                                  output_model_file,
                                  processor,
                                  label_list,
                                  tokenizer,
                                  device,
                                  n_gpu,
                                  tensorboard_logger,
                                  eval_data)

    model_config = None
    model_embeddings = None
    if args.onnx_framework is None:
        # Load a trained model that you have fine-tuned
        if os.path.exists(output_model_file):
            model_state_dict = torch.load(output_model_file, map_location=lambda storage, loc: storage)
        else:
            model_state_dict = None
        model = BertForSequenceClassification.from_pretrained(args.bert_model,
                                                              state_dict=model_state_dict,
                                                              num_labels=num_labels)
        model_config = copy.deepcopy(model.config)
        model_embeddings = model.bert.embeddings
        model = convert_model(args, model, device, n_gpu)
    else:
        import onnx
        model = onnx.load(onnx_model_file)
        onnx.checker.check_model(model)

    if args.do_distill:
        assert model_config is not None
        assert model_embeddings is not None
        output_distilled_model_file = os.path.join(args.output_dir, DISTILLER_WEIGHTS_NAME)
        teacher = model
        model_config.hidden_act = args.blendcnn_act
        if args.blendcnn_pair:
            student = BlendCNNForSequencePairClassification(model_config,
                                                            num_labels=num_labels,
                                                            channels=(model_config.hidden_size,) +
                                                                     args.blendcnn_channels,
                                                            n_hidden_dense=(model_config.hidden_size,) * 2,
                                                            use_dropout=args.blendcnn_dropout)
        else:
            student = BlendCNN(model_config,
                               num_labels=num_labels,
                               channels=(model_config.hidden_size,) + args.blendcnn_channels,
                               n_hidden_dense=(model_config.hidden_size,) * 2,
                               use_dropout=args.blendcnn_dropout)
        student.embeddings.load_state_dict(model_embeddings.state_dict())

        student = convert_model(args, student, device, 1)
        if os.path.exists(output_distilled_model_file):
            logger.info(
                'Loading existing distilled model {}, skipping distillation'.format(output_distilled_model_file))
            student.load_state_dict(torch.load(output_distilled_model_file))
        else:
            dlw = distiller.DistillationLossWeights(args.kd_distill_wt, args.kd_student_wt, args.kd_teacher_wt)
            args.kd_policy = distiller.KnowledgeDistillationPolicy(student, teacher, args.kd_temp, dlw)

            tensorboard_log_dir = os.path.join(args.output_dir, './log')
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            tensorboard_logger = SummaryWriter(tensorboard_log_dir)

            if args.do_eval and do_eval_or_test(args) and eval_data is None:
                eval_data = prepare(args, processor, label_list, tokenizer, 'dev')

            global_step, loss = distill(args,
                                        output_distilled_model_file,
                                        processor,
                                        label_list,
                                        tokenizer,
                                        device,
                                        n_gpu,
                                        tensorboard_logger,
                                        eval_data)
        model = student

    if do_eval_or_test(args):
        result = {
            'global_step': global_step,
            'loss': loss
        }
        model.float()
        name = '_distiller' if args.do_distill else ''

        if args.do_eval:
            if eval_data is None:
                eval_data = prepare(args, processor, label_list, tokenizer, 'dev')
            eval_loss, eval_accuracy, eval_probs = eval(args, model, eval_data, device, verbose=True)
            np.savetxt(os.path.join(args.output_dir, 'dev{}_probs.npy'.format(name)), eval_probs)
            result.update({
                'dev{}_loss'.format(name): eval_loss,
                'dev{}_accuracy'.format(name): eval_accuracy,
            })

        if args.do_test:
            eval_data = prepare(args, processor, label_list, tokenizer, 'test')
            eval_loss, eval_accuracy, eval_probs = eval(args, model, eval_data, device, verbose=True)
            np.savetxt(os.path.join(args.output_dir, 'test{}_probs.npy'.format(name)), eval_probs)
            result.update({
                'test{}_loss'.format(name): eval_loss,
                'test{}_accuracy'.format(name): eval_accuracy,
            })

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

    if args.export_onnx:
        if not env_enabled(ENV_OPENAIGPT_GELU) or not env_enabled(ENV_DISABLE_APEX):
            raise ValueError('Both {} and {} must be 1 to properly export ONNX.'.format(ENV_OPENAIGPT_GELU,
                                                                                        ENV_DISABLE_APEX))

        if not isinstance(model, torch.nn.Module):
            raise ValueError('model is not an instance of torch.nn.Module.')

        import onnx
        import onnx.utils
        import onnx.optimizer
        dummy_input = get_dummy_input(args, processor, label_list, tokenizer, device)
        torch.onnx.export(model,
                          dummy_input,
                          onnx_model_file,
                          input_names=['input_ids', 'input_mask', 'segment_ids'],
                          output_names=['output_logit'],
                          verbose=True)
        optimized_model = onnx.optimizer.optimize(onnx.load(onnx_model_file),
                                                  [pass_ for pass_ in onnx.optimizer.get_available_passes()
                                                   if 'split' not in pass_])
        optimized_model = onnx.utils.polish_model(optimized_model)
        onnx.save(optimized_model, os.path.join(args.output_dir, 'optimized_model.onnx'))


def get_dummy_input(args, processor, label_list, tokenizer, device):
    dummy_features = convert_examples_to_features(
        [InputExample(guid='dummy-0', text_a=' ', text_b=' ',
                      label=processor.labels[0])],
        label_list,
        args.max_seq_length,
        tokenizer)
    dummy_input_ids = torch.tensor([f.input_ids for f in dummy_features], dtype=torch.long).to(device)
    dummy_input_mask = torch.tensor([f.input_mask for f in dummy_features], dtype=torch.long).to(device)
    dummy_segment_ids = torch.tensor([f.segment_ids for f in dummy_features], dtype=torch.long).to(device)
    dummy_input = (dummy_input_ids, dummy_segment_ids, dummy_input_mask)
    return dummy_input


def do_eval_or_test(args):
    return (args.do_eval or args.do_test) and (args.local_rank == -1 or torch.distributed.get_rank() == 0)


def convert_model(args, model, device, n_gpu):
    if not args.no_cuda:
        if args.fp16:
            model.half()
        model.to(device)
        if args.local_rank != -1:
            try:
                from apex.parallel import DistributedDataParallel as DDP
            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

            model = DDP(model)
        elif n_gpu > 1:
            model = torch.nn.DataParallel(model)

    return model


def get_optimizer(args, model, num_train_steps):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    t_total = num_train_steps
    if args.local_rank != -1:
        t_total = t_total // torch.distributed.get_world_size()
    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)

    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=t_total)

    return optimizer, t_total


def save_model(model, output_model_file):
    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
    torch.save(model_to_save.state_dict(), output_model_file)


def train(args,
          model,
          output_model_file,
          processor,
          label_list,
          tokenizer,
          device,
          n_gpu,
          tensorboard_logger,
          eval_data=None):
    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    save_best_model = eval_data is not None and args.eval_interval > 0

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    optimizer, t_total = get_optimizer(args, model, num_train_steps)

    train_data = prepare(args, processor, label_list, tokenizer, 'train')
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    train_steps = 0
    best_eval_accuracy = 0
    for _ in trange(int(args.num_train_epochs), desc="Epoch", dynamic_ncols=True):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            model.train()
            loss = model(input_ids, segment_ids, input_mask, label_ids)
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            train_steps += 1
            tensorboard_logger.add_scalar('train_loss', loss.item(), train_steps)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if save_best_model and train_steps % args.eval_interval == 0:
                eval_loss, eval_accuracy, _ = eval(args, model, eval_data, device, verbose=False)
                tensorboard_logger.add_scalar('dev_loss', eval_loss, train_steps)
                tensorboard_logger.add_scalar('dev_accuracy', eval_accuracy, train_steps)
                if eval_accuracy > best_eval_accuracy:
                    save_model(model, output_model_file)
                    best_eval_accuracy = eval_accuracy

    if save_best_model:
        eval_loss, eval_accuracy, _ = eval(args, model, eval_data, device, verbose=False)
        if eval_accuracy > best_eval_accuracy:
            save_model(model, output_model_file)
    else:
        save_model(model, output_model_file)

    return global_step, tr_loss / nb_tr_steps


def prepare(args, processor, label_list, tokenizer, task):
    if task == 'train':
        eval_examples = processor.get_train_examples(args.data_dir)
    elif task == 'dev':
        eval_examples = processor.get_dev_examples(args.data_dir)
    elif task == 'test':
        eval_examples = processor.get_test_examples(args.data_dir)
    else:
        raise NotImplementedError(task)

    features = convert_examples_to_features(
        eval_examples,
        label_list,
        args.max_seq_length,
        tokenizer,
        num_workers=os.cpu_count() // 2,
        desc='Convert {} examples'.format(task),
        verbose=True)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    return TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)


def eval(args, model, eval_data, device, verbose=False):
    if not isinstance(model, torch.nn.Module):
        return onnx_eval(args, model, eval_data, verbose=verbose)

    # Run prediction for full data
    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = []

    model.eval()

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                              desc="Evaluating",
                                                              leave=verbose,
                                                              dynamic_ncols=True):
        input_ids = input_ids.to(device)
        input_mask = input_mask.to(device)
        segment_ids = segment_ids.to(device)
        label_ids = label_ids.to(device)

        with torch.no_grad():
            tmp_eval_loss = model(input_ids, segment_ids, input_mask, label_ids)
            logits = model(input_ids, segment_ids, input_mask)

        logits = logits.detach().cpu()
        label_ids = label_ids.to('cpu').numpy()
        tmp_eval_accuracy = accuracy(logits.numpy(), label_ids)

        if verbose:
            all_logits.append(logits)

        eval_loss += tmp_eval_loss.mean().item()
        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_examples
    eval_probs = None
    if all_logits:
        eval_probs = F.softmax(torch.cat(all_logits), dim=-1).numpy()

    return eval_loss, eval_accuracy, eval_probs


def distill(args,
            output_model_file,
            processor,
            label_list,
            tokenizer,
            device,
            n_gpu,
            tensorboard_logger,
            eval_data=None):
    assert args.kd_policy is not None
    model = args.kd_policy.student
    args.kd_policy.teacher.eval()
    num_labels = len(args.labels)

    global_step = 0
    nb_tr_steps = 0
    tr_loss = 0
    save_best_model = eval_data is not None and args.eval_interval > 0

    train_examples = processor.get_train_examples(args.data_dir)
    num_train_steps = int(
        len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    optimizer, t_total = get_optimizer(args, model, num_train_steps)

    train_data = prepare(args, processor, label_list, tokenizer, 'train')
    logger.info("***** Running distillation *****")
    logger.info("  Num examples = %d", len(train_examples))
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_steps)

    if args.local_rank == -1:
        train_sampler = RandomSampler(train_data)
    else:
        train_sampler = DistributedSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    train_steps = 0
    best_eval_accuracy = 0
    for epoch in trange(int(args.num_train_epochs), desc="Epoch", dynamic_ncols=True):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        args.kd_policy.on_epoch_begin(model, None, None)

        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", dynamic_ncols=True)):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            model.train()
            logits = args.kd_policy.forward(input_ids, segment_ids, input_mask)
            loss = CrossEntropyLoss()(logits.view(-1, num_labels), label_ids.view(-1))
            loss = args.kd_policy.before_backward_pass(model, epoch, None, None, loss, None).overall_loss
            if n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu.
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                optimizer.backward(loss)
            else:
                loss.backward()

            train_steps += 1
            tensorboard_logger.add_scalar('distillation_train_loss', loss.item(), train_steps)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # modify learning rate with special warm up BERT uses
                lr_this_step = args.learning_rate * warmup_linear(global_step / t_total, args.warmup_proportion)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_this_step
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

            if save_best_model and train_steps % args.eval_interval == 0:
                eval_loss, eval_accuracy, _ = eval(args, model, eval_data, device, verbose=False)
                tensorboard_logger.add_scalar('distillation_dev_loss', eval_loss, train_steps)
                tensorboard_logger.add_scalar('distillation_dev_accuracy', eval_accuracy, train_steps)
                if eval_accuracy > best_eval_accuracy:
                    save_model(model, output_model_file)
                    best_eval_accuracy = eval_accuracy

        args.kd_policy.on_epoch_end(model, None, None)

    if save_best_model:
        eval_loss, eval_accuracy, _ = eval(args, model, eval_data, device, verbose=False)
        if eval_accuracy > best_eval_accuracy:
            save_model(model, output_model_file)
    else:
        save_model(model, output_model_file)

    return global_step, tr_loss / nb_tr_steps


def onnx_eval(args, onnx_model, eval_data, verbose=False):
    # Run prediction for full data
    args.eval_batch_size = 1
    if verbose:
        logger.info("***** Running evaluation *****")
        logger.info("  Num examples = %d", len(eval_data))
        logger.info("  Batch size = %d", args.eval_batch_size)
    eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=args.eval_batch_size)

    eval_loss, eval_accuracy = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    all_logits = []

    if args.onnx_framework == 'caffe2':
        import caffe2.python.onnx.backend as backend
        prepared_backend = backend.prepare(onnx_model)
        model = lambda x, y, z: prepared_backend.run((x, y, z))
    else:
        raise NotImplementedError(args.framework)

    for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader,
                                                              desc="Evaluating",
                                                              leave=verbose,
                                                              dynamic_ncols=True):
        input_ids = input_ids.numpy()
        input_mask = input_mask.numpy()
        segment_ids = segment_ids.numpy()
        label_ids = label_ids.numpy()

        logits = model(input_ids, segment_ids, input_mask)

        tmp_eval_accuracy = accuracy(logits, label_ids)

        if verbose:
            all_logits.append(logits)

        eval_accuracy += tmp_eval_accuracy

        nb_eval_examples += input_ids.shape[0]
        nb_eval_steps += 1

    eval_accuracy = eval_accuracy / nb_eval_examples
    eval_probs = None
    if all_logits:
        eval_probs = np_softmax(np.concatenate(all_logits)).squeeze()

    return None, eval_accuracy, eval_probs


def np_softmax(x, t=1):
    x = x / t
    x = x - np.max(x, axis=-1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=-1, keepdims=True)


if __name__ == "__main__":
    main()
