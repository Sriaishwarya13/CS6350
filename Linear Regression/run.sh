#!/bin/bash

echo Batch Gradient Descent
python3 LMS.py batch
echo ____________________________
echo Stochastic Gradient Descent
python3 LMS.py stoch
