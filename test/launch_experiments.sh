datas=(toy_data_001)

seeds=(2628670643 2628670647)

# Loop through each prompt path
for data in "${datas[@]}"; do
 echo "Running with prompt_path=$data"
 python InterActing/prompt_completion.py --data_dir InterActing/data \
 --scenario toy_data_001 --type decomposition

 # Loop through each seed
 for seed in "${seeds[@]}"; do
   echo "Running with seed=$seed"
   CUDA_VISIBLE_DEVICES=1 python DetailScribe/DetailScribe.py \
     --prompt_path="InterActing/data/decomposition/$data.jsonl" \
     --seed="$seed" \
     --experiment_name init_only
 done
done

experiment_names=(gpt_refine gpt_rewrite)

for data in "${datas[@]}"; do
 echo "Running with prompt_path=$data"
 # Loop through each seed
 CUDA_VISIBLE_DEVICES=1 python DetailScribe/DetailScribe.py \
   --prompt_path="InterActing/data/decomposition/$data.jsonl" \
   --catlog_path="results/$data/catlogs_2628670643.jsonl" \
   --experiment_name DetailScribe
 CUDA_VISIBLE_DEVICES=1 python DetailScribe/DetailScribe.py \
     --new_prompt_path="results/$data/new_prompt_2628670643_detailscribe.jsonl" \
     --experiment_name dalle3
 for ename in "${experiment_names[@]}"; do
   echo "Running experiment=$ename"
   CUDA_VISIBLE_DEVICES=1 python DetailScribe/DetailScribe.py \
     --catlog_path="results/$data/catlogs_2628670643.jsonl" \
     --experiment_name="$ename"
 done
done