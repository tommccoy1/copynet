
import argparse

from data_loading import *
from training import *
from utils import * 

parser = argparse.ArgumentParser()
parser.add_argument("--model", help="model type (transformer, tp_transformer, gru, lstm)", type=str, default=None)
parser.add_argument("--dataset_prefix", help="prefix for the training, validation, and test dataset", type=str, default=None)
parser.add_argument("--patience", help="number of checkpoints without improvement to wait before early stopping", type=int, default=5)
parser.add_argument("--eval_every", help="number of batches to train on in between each evaluation", type=int, default=1000)
parser.add_argument("--batch_size", help="batch size", type=int, default=1)
parser.add_argument("--hidden", help="hidden size", type=int, default=256)
parser.add_argument("--filter", help="feedforward dimensions", type=int, default=1024)
parser.add_argument("--dropout", help="dropout percentage", type=float, default=0.1)
parser.add_argument("--max_length", help="maximum sequence length", type=int, default=20)
parser.add_argument("--n_layers", help="number of layers", type=int, default=2)
parser.add_argument("--n_heads", help="number of attention heads", type=int, default=4)
parser.add_argument("--vocab_size", help="vocabulary size", type=int, default=10)
parser.add_argument("--training_size", help="number of training examples to use", type=int, default=None)
parser.add_argument("--n_epochs", help="maximum number of training epochs", type=int, default=1000)
parser.add_argument("--model_name", help="name for saving model weights", type=str, default=None)
parser.add_argument("--random_seed", help="random seed", type=int, default=None)
parser.add_argument("--print_errors", help="Print all errors on the gen set", action='store_true')
args = parser.parse_args()

print(args)

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

# Train model
model.train()
train(model, training_set=training_set, valid_set=valid_set, patience=args.patience, eval_every=args.eval_every, sos_token=args.vocab_size, eos_token=args.vocab_size+1, n_epochs=args.n_epochs, model_name=args.model_name)


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

# Evaluate on test set
first = True

correct = 0
total = 0
for elt in test_set:
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

        if filtered == right_answer:
            correct += 1
        total += 1

print("Test set:", correct * 1.0 / total, correct, total)


# Evaluate on generalization set
first = True

correct = 0
total = 0
for elt in gen_set:
    preds = model.greedy_inference(torch.LongTensor(elt), args.vocab_size, args.vocab_size+1, 20, "cpu").tolist()

    if first:
        print("Gen examples")
        print(elt)
        print([filter_greedy(pred) for pred in preds])
        print("")
        first = False

    for index, pred in enumerate(preds):
        filtered = filter_greedy(pred)
        
        right_answer = elt[index]

        if filtered == right_answer:
            correct += 1
        else:
            if args.print_errors:
                print(right_answer)
                print(filtered)
                print("")
        total += 1

if total != 0:
    print("Gen set:", correct * 1.0 / total, correct, total)
else:
    print("Gen set:", 0.0, correct, total)

# Print the parameter count
pytorch_total_params = sum(p.numel() for p in model.parameters())
print("Parameter count:", pytorch_total_params)
