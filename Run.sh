#!/bin/bash

# This is a script to run all relevant code for the part of speech tagger.

echo ""
echo "Estimating weight vector with perceptron algorithm and creating tagger.model..."
echo "estimated run time: ~ 50 seconds"
python perceptron.py data/tag_train.dat > tagger.model

echo ""
echo "Tagging development data using new features and writing to tag_dev.out..."
python tagger.py tagger.model data/tag_dev.dat > tag_dev.out

echo ""
echo "Evaluating tagger..."
python lib/eval_tagger.py data/tag_dev.key tag_dev.out
