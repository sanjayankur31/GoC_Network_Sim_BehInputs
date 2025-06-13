#!/usr/bin/env python3
"""
Print contents of a Pickled file

File: print_pickle.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import pickle
import sys

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("One argument required: pickle file")
        sys.exit(-1)

    print(f"Printing contents of {sys.argv[1]}\n")
    with open(sys.argv[1], "rb") as f:
        data = pickle.load(f, encoding="bytes")
        print(data)
