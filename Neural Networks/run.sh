#!/bin/bash

echo "running testing net"
python3 test.py

echo "running nn with random weight"
python3 init.py

echo "running nn with zeros weight"
python3 zero.py

echo "running pytorch,2 e"
python3 pytorch_2e.py
