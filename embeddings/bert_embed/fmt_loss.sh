#!/bin/bash

pair_num=0
t=""
v=""
while read line; do
  if [[ "$line" =~ ^Train.+loss.+([0-9]\.[0-9]+).+$ ]]; then
    train_loss="${BASH_REMATCH[1]}"
    t+="($((++pair_num)),$train_loss),"
  elif [[ "$line" =~ ^Valid.+loss.+([0-9]\.[0-9]+).+$ ]]; then 
    val_loss="${BASH_REMATCH[1]}"
    v+="($((pair_num)),$val_loss),"
  fi
done < ./transformer_bert_loss.txt

echo $t
echo $v
