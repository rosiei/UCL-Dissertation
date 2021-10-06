
################################


CUDA_VISIBLE_DEVICES="cranky_liskov" python bert_influence.py --output_dir="influence_func_output" --data_dir="/project/my_data"\
    --bert_model="bert-base-uncased" --do_lower_case --trained_model_dir="/project/my_model/model-runner-output" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2021\
    --damping=3e-3 --scale=1e8 --lissa_repeat=1 --lissa_depth_pct=0.25\
    --logging_steps=200 --full_bert --alt_mode="dirctr"



CUDA_VISIBLE_DEVICES="cranky_liskov" python bert_trackin.py --output_dir="tracin_func_output" --data_dir="/project/my_data"\
    --bert_model="bert-base-uncased" --do_lower_case --trained_model_dir="/project/my_model/model-runner-output" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2021\
    --logging_steps=200 --full_bert  --alt_mode="dirctr"

