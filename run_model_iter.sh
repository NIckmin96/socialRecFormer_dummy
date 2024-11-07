#!/bin/bash

for i in {7..7..1}
do
	echo "Running $i times training dataset model..." >> training_log.txt
	python main.py --dataset ciao --train_augs $i --test_augs True >> training_log.txt
	
done

