
import argparse

from data_loading import *
from training import *
from utils import * 

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model type (transformer, tp_transformer)", type=str, default=None)
parser.add_argument("--dataset_prefix", help="prefix for the training, validation, and test dataset", type=str, default=None)
parser.add_argument("--patience", help="number of checkpoints without improvement to wait before early stopping", type=int, default=5)
parser.add_argument("--eval_every", help="number of batches to train on in between each evaluation", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=1)
parser.add_argument("--hidden", help="hidden size", type=int, default=256)
parser.add_argument("--filter", help="feedforward dimensions", type=int, default=1024)
parser.add_argument("--dropout", help="dropout percentage", type=float, default=0.1)
parser.add_argument("--max_length", help="maximum sequence length", type=int, default=20)
parser.add_argument("--n_layers", help="number of layers", type=int, default=2)
parser.add_argument("--n_heads", help="number of attention heads", type=int, default=8)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=10)
parser.add_argument("--training_size", help="number of training examples to use", type=int, default=None)
parser.add_argument("--n_epochs", help="maximum number of training epochs", type=int, default=1000)
parser.add_argument("--model_name", help="name for saving model weights", type=str, default=None)
parser.add_argument("--random_seed", help="random seed", type=int, default=None)
parser.add_argument('--all_data_list', help="exhaustive list of all sequences", type=str, default=None)
parser.add_argument("--print_errors", help="Print all errors", action='store_true')

args = parser.parse_args()

print(args)


def insertion(seq1, seq2):
    if len(seq1) == 1 + len(seq2):
        for i in range(len(seq1)):
            new_seq = seq1[:i] + seq1[i+1:]
            if new_seq == seq2:
                return True
    return False

def deletion(seq1, seq2):
    return insertion(seq2, seq1)

def substitution(seq1, seq2):
    if len(seq1) == len(seq2):
        count_diff = 0
        for elt1, elt2 in zip(seq1, seq2):
            if elt1 != elt2:
                count_diff += 1

        if count_diff == 1:
            return True
    return False

def swap(seq1, seq2):
    if len(seq1) == len(seq2):
        diff_pairs = []
        for elt1, elt2 in zip(seq1, seq2):
            if elt1 != elt2:
                diff_pairs.append([elt1, elt2])
        if len(diff_pairs) == 2:
            if diff_pairs[0] == diff_pairs[1][::-1]:
                return True

    return False

def classify_output(seq1, seq2):
    if seq1 == seq2:
        return "correct"
    elif insertion(seq1, seq2):
        return "insertion"
    elif deletion(seq1, seq2):
        return "deletion"
    elif substitution(seq1, seq2):
        return "substitution"
    elif swap(seq1, seq2):
        return "swap"
    else:
        return "other"


# Set random seed
if args.random_seed is not None:
    random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)


# Create model
if args.model == "transformer":
    from transformer import *
elif args.model == "tp_transformer":
    from tp_transformer import *
elif args.model == "lstm" or args.model == "gru":
    from rnn import *

if "transformer" in args.model:
    # pad_idx is a dummy, we aren't using it
    # We can get away without padding because all of our sequences
    # are the same length. Padding would need to be implemented
    # for more complex tasks
    model = build_transformer(input_dim=args.vocab_size+2, hidden=args.hidden, dropout=args.dropout, max_length=args.max_length, n_layers=args.n_layers, n_heads=args.n_heads, myfilter=args.filter, pad_idx=1234567890)
else:
    model = RNNSeq2Seq(args.hidden, args.vocab_size+2, args.model.upper(), n_layers=args.n_layers, dropout_p=args.dropout, sos_token=args.vocab_size)


# Load data
training_set = load_dataset("data/" + args.dataset_prefix + ".train")

if args.training_size is not None:
    training_set = training_set[:args.training_size]

training_set = batchify(training_set, args.batch_size)
valid_set = batchify(load_dataset("data/" + args.dataset_prefix + ".valid"), args.batch_size)
test_set = batchify(load_dataset("data/" + args.dataset_prefix + ".test"), args.batch_size)
gen_set = batchify(load_dataset("data/" + args.dataset_prefix + ".gen"), args.batch_size) 

all_eval_set = batchify(load_dataset("data/" + args.all_data_list), args.batch_size)

# Evaluate best checkpoint
model.load_state_dict(torch.load("weights/" + args.model_name + ".weights"))

# Trim an output sequence to end at its EOS token
def filter_greedy(seq):
    new_seq = []

    for elt in seq[1:]:
        if elt == args.vocab_size+1:
            return new_seq
        else:
            new_seq.append(elt)

    return new_seq


def count_deviations(seq):
    deviations = 0

    for index, elt in enumerate(seq):
        if index == elt:
            deviations += 1

    return deviations

# Evaluate on test set
first = True

correct_dict = {}
output_categories = {}
category_list = ["correct", "insertion", "deletion", "substitution", "swap", "other"]
for category in category_list:
    output_categories[category] = 0

for eval_set_index, elt in enumerate(all_eval_set):
    if eval_set_index % 1000 == 0:
        print(eval_set_index, len(all_eval_set))
    preds = model.greedy_inference(torch.LongTensor(elt), args.vocab_size, args.vocab_size+1, 20, "cpu").tolist()

    # Print a few examples of test sequences
    # and model predictions
    if first:
        print("Test examples")
        print(elt)
        print([filter_greedy(pred) for pred in preds])
        print("")
        first = False

    for index, pred in enumerate(preds):
        filtered = filter_greedy(pred)
       
        right_answer = elt[index]

        deviations = count_deviations(right_answer)
        if deviations not in correct_dict:
            correct_dict[deviations] = [0,0]

        category = classify_output(filtered, right_answer)
        output_categories[category] += 1

        if filtered == right_answer:
            correct_dict[deviations][0] += 1
        else:
            if args.print_errors:
                print(category)
                print(right_answer)
                print(filtered)
                print("")
        correct_dict[deviations][1] += 1


print("Eval results:")

done = False
deviation_count = 0

while not done:
    if deviation_count not in correct_dict:
        done = True
    else:
        correct = correct_dict[deviation_count][0]
        total = correct_dict[deviation_count][1]
        print(str(deviation_count) + " deviations:", correct*1.0/total, correct, total)
        deviation_count += 1

print("")
for category in output_categories:
    print(category, output_categories[category])

print("")

# Print the parameter count
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Parameter count:", pytorch_total_params)
