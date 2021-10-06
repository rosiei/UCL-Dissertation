CUDA_VISIBLE_DEVICES="big_feynman" python xlnet_influence.py --output_dir="influence_func_output" --data_dir="/project/my_data"\
    --xlnet_model="xlnet-base-cased" --do_lower_case --trained_model_dir="/project/XLNet_model/model-runner-output2" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2021\
    --damping=3e-3 --scale=1e8 --lissa_repeat=1 --lissa_depth_pct=0.25\
    --logging_steps=200 --alt_mode="dirctr"



CUDA_VISIBLE_DEVICES="big_feynman" python xlnet_trackin.py --output_dir="tracin_func_output" --data_dir="/project/my_data"\
    --xlnet_model="xlnet-base-cased" --do_lower_case --trained_model_dir="/project/XLNet_model/model-runner-output2" --max_seq_length=200\
    --train_batch_size=8 --eval_batch_size=1 --seed=2021\
    --logging_steps=200 --alt_mode="dirctr" --num_trained_epoch=3

