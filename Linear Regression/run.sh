#!/bin/bash

echo Batch Gradient Descent
python3 LMS.py bgd
echo ____________________________
echo Stochastic Gradient Descent
python3 LMS.py sgd
