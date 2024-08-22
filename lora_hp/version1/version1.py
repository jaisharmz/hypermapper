#!/usr/bin/python
import math

import os
import sys
import warnings
from collections import OrderedDict

from hypermapper import optimizer  # noqa

from lora_run_experiment import get_rouge_score


def obj_function(X):
    return get_rouge_score(X["learning_rate"], X["rank"])


def main():
    parameters_file = "lora_hp_scenario.json"
    optimizer.optimize(parameters_file, obj_function)
    print("End of LoRA finetuning.")


if __name__ == "__main__":
    main()
