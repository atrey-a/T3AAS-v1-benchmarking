import sys
from tqdm import tqdm

def progress_bar(is_bar, exp_name):
    if is_bar:
        print(f"Running Experiment: {exp_name}", file=sys.__stderr__)
        def pb(iterator, *args, **kwargs):
            return tqdm(iterator, desc="Epochs", *args, **kwargs)
        return pb
    else:
        def pbn(iterator):
            return iterator
        return pbn
