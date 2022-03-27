import logging
import os
import random
import shutil

import torch


class HyperParams:
  def __init__(self, **kwargs):
    self.__dict__.update(kwargs)

  def keys(self):
    return self.__dict__.keys()

  def __repr__(self):
    keys = self.__dict__.keys()
    items = ("{}={!r}".format(k, self.__dict__[k]) for k in keys)
    return "{}:\n\t{}".format(type(self).__name__, "\n\t".join(items))

  def __eq__(self, other):
    return self.__dict__ == other.__dict__

def batchify(dataset, batch_size):
    batched = []

    this_batch = []
    for elt in dataset:
        this_batch.append(elt)

        if len(this_batch) == batch_size:
            batched.append(this_batch[:])
            this_batch = []

    if len(this_batch) > 0:
        batched.append(this_batch[:])

    return batched




 
