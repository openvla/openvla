OpenVLA but with `finetune.py` using a `LeRobotDataset` instead of a `RLDSDataset`. 

To use:
1. Make a directory `data`: `cd openvla; mkdir data`
2. Put your dataset in LeRobot format inside that directory.
3. Run `finetune.py`, for example, like this. First line is to add the `openvla/lerobot` submodule to Python search path. This way of loading `lerobot` helped me avoid some dependency conflicts. 
```bash
PYTHONPATH="$PYTHONPATH:$(pwd)/lerobot" \
torchrun --standalone --nnodes 1 --nproc-per-node 1 vla-scripts/finetune.py --vla_path "openvla/openvla-7b" --data_root_dir "data" --dataset_name converted_libero_v6 --run_root_dir .runs --adapter_tmp_dir .adapter --lora_rank 32 --batch_size 2 --grad_accumulation_steps 8 --learning_rate 5e-4 --image_aug True --save_steps 10
```
Here we assume you have a dataset called `converted_libero_v6` inside `data`. 
