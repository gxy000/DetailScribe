# call data-4o api to generate prompts from the list of topics (interactions)
python pr/prompt_gen_gpt.py --scenario tool_manipulation_150_gpt

# call data-4o api to decompose the interactions in the prompts
python pr/prompt_gen_gpt.py --scenario tool_manipulation_150_gpt --type decomposition

# distinct duplicated topics
python pr/dup_check.py \
--input data/decomposition/tool_manipulation_150_gpt.jsonl --type decomposition



python InterActing/prompt_gen_gpt.py --scenario toy_data_1 --type decomposition