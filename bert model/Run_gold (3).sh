CUDA_VISIBLE_DEVICES='gigantic_swanson' python -W ignore bert_runner.py --data_dir="/project/my_data"  --bert_model="bert-base-uncased" --max_seq_length=200 --do_train --do_test --train_batch_size=24 --eval_batch_size=1 --learning_rate=5e-5 --num_train_epochs=3 --warmup_proportion=0.1 --seed=2020 --output_dir="Gold_model_output" --do_lower_case --full_bert --mode="dirctr_gold_train" 

