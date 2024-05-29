#!/bin/bash

srun -n 12 --gpu-bind=map_gpu:0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3 mace-md -f /path/to/input_file.pdb\
  --ml_mol /path/to/input_file.pdb \
  --resname "LIG"\
  --output_dir "./16_reps_production" \
  --pressure 1.0 \
  --decouple \
  --restart \
  --replicas 16 \
  --log_level DEBUG \
  --nl "nnpops" \
  --run_type "repex" \
  --system_type "pure" \
  --dtype float64 \
  --interval 1 \
  --steps 5000 \
  --model_path "/path/to/MACE.model" \
  --steps_per_iter 1000 \
  --lambda_schedule "[0.,   0.04, 0.08     , 0.12, 0.16,   0.20, 0.2425, 0.28, 0.35, 0.4 , 0.52, 0.64, 0.76, 0.88, 0.95, 1.   ]" \
