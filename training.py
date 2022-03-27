import torch
import torch.nn as nn
from torch import optim
import math

def train(model, training_set=None, valid_set=None, n_epochs=100, patience=5, stopping_criterion="loss", eval_every=1000, sos_token=None, eos_token=None, task="copy", model_name=None):
    best_valid_acc = -1
    best_valid_loss = math.inf
    checkpoints_since_improved = 0


    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = nn.NLLLoss()
    done = False
    
    for epoch in range(n_epochs):
        if done:
            break
        for index in range(len(training_set)):
            optimizer.zero_grad()
            elt = training_set[index % len(training_set)] # Choosing batch to train on

            # Determine what the correct output sequence should be
            correct = [x + [eos_token] for x in elt]
            logits = nn.LogSoftmax(dim=2)(model(torch.LongTensor(elt), torch.LongTensor([[sos_token] + x for x in elt])))

            topv, topi = logits.topk(1)
            preds = topi.squeeze(2).tolist()


            loss = criterion(logits.transpose(1,2), torch.LongTensor(correct))
            loss.backward()
            optimizer.step()   

            if index % eval_every == 0:

                right = 0
                total = 0

                valid_loss = 0

                for elt in valid_set:
                    correct = [x + [eos_token] for x in elt]
                    logits = nn.LogSoftmax(dim=2)(model(torch.LongTensor(elt), torch.LongTensor([[sos_token] + x for x in elt])))

                    topv, topi = logits.topk(1)
                    preds = topi.squeeze(2).tolist()

                    for guess, answer in zip(preds, correct):
                        if guess == answer:
                            right += 1
                        total += 1

                    loss = criterion(logits.transpose(1,2), torch.LongTensor(correct))

                    valid_loss += loss

                valid_acc = right*1.0 / total
                print(valid_acc, valid_loss.item())
                if stopping_criterion == "acc":

                    if valid_acc > best_valid_acc:
                        best_valid_acc = valid_acc
                        checkpoints_since_improved = 0
                        torch.save(model.state_dict(), "weights/" + model_name + ".weights")
                    else:
                        checkpoints_since_improved += 1
                        
        
                elif stopping_criterion == "loss":
                    if valid_loss < best_valid_loss:
                        best_valid_loss = valid_loss
                        checkpoints_since_improved = 0
                        torch.save(model.state_dict(), "weights/" + model_name + ".weights")
                    else:
                        checkpoints_since_improved += 1

                if checkpoints_since_improved >= patience:
                    done = True
                    break


