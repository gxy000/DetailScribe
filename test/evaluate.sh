metrics=(ImageReward CLIPScore)
#metrics=(gpt)

data_names=(toy_data_001)

 for i in "${!data_names[@]}"; do
   echo "Running with:"
   echo "  data_name=${data_names[$i]}"

    CUDA_VISIBLE_DEVICES=0 python DetailScribe/results_cleanup_and_preprocess.py \
    --new_prompt_path "results/${data_names[$i]}/new_prompt_2628670643_detailscribe.jsonl" \
    --primary_seed 2628670643 --multi_seed 2628670647

   for j in "${!metrics[@]}"; do
     echo "evaluating metric=${metrics[$j]}"
     CUDA_VISIBLE_DEVICES=0 python DetailScribe/auto_evaluators.py \
     "results/eval/dict/main_${data_names[$i]}.jsonl" \
     --metric ${metrics[$j]} \
     --by_scale
   done
 done