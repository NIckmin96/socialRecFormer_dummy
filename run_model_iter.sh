#!/bin/bash

for i in {5..10..1}
do
	echo "Running $i times training dataset model..." >> training_log.txt
	python main.py --dataset ciao --train_augs $i >> training_log.txt
	
done

