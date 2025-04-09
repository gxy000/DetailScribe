from email.policy import default

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
from pathlib import Path
import os
import json
import time
import base64
import re
import requests

from openai import OpenAI

MMDD = time.strftime("%m%d%H")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def read_jsonl(path):
    with open(path, 'r') as f:
        dicts = [json.loads(line) for line in f]
    dicts = [json.loads(x) if isinstance(x, str) else x for x in dicts]
    return dicts


def write_jsonl(contents, path):
    with open(path, 'w') as f:
        for each in contents:
            f.write(json.dumps(each) + '\n')


def run_original(pipeline, data, parent_dir, seed, save_noise = False):
    topic = data['topic']
    out_dir = f"{parent_dir}/{topic}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_path = f'{out_dir}/original_{seed}.jpeg'
    if 'prompt' in data.keys():
        prompt = data['prompt']
    else:
        prompt = data['original_prompt']

    image = pipeline(
        prompt=prompt,
        num_inference_steps=28,
        guidance_scale=4.5,
        max_sequence_length=512,
        generator=torch.Generator(device="cpu").manual_seed(seed),
    ).images[0]

    image.save(out_path)
    print(f'image saved at {out_path}')
    return out_dir


def inference_pos(pipeline, prompts, out_dir, prompts_type, max_distance, seed, step_size=0.1, save_intermediate=[]):
    out_dir = f"{out_dir}/{prompts_type}"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    print(f'saved in {out_dir}')
    if not isinstance(max_distance, list):
        max_distance = range(max_distance)
    for switch_time in max_distance:
        def decode_tensors(pipe, step, timestep, callback_kwargs):
            if switch_time in save_intermediate:
                latents = callback_kwargs["latents"]
                intermediate_dir = f'{out_dir}/{switch_time}'
                Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
                # torch.save(latents, f"{latent_dir}/tensor_{step:04d}.pt")

                latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
                if step % 2 == 0:
                    image = pipe.vae.decode(latents, return_dict=False)[0]
                    image = pipe.image_processor.postprocess(image, output_type="pil")
                    image[0].save(f"{intermediate_dir}/image_{step:04d}.jpeg")

            switch_step = int(switch_time) if step_size >= 1 else int(step_size * switch_time * pipe.num_timesteps)
            if step == switch_step:
                print(f"At step {step}, update prompt to {prompts[1]}")
                lora_scale = (
                    pipe.joint_attention_kwargs.get("scale", None) if pipe.joint_attention_kwargs is not None else None
                )
                prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                    prompt=prompts[1],
                    prompt_2=None,
                    prompt_3=None,
                    do_classifier_free_guidance=pipe.do_classifier_free_guidance,
                    device=pipe._execution_device,
                    clip_skip=pipe.clip_skip,
                    lora_scale=lora_scale,
                    max_sequence_length=512,
                )
                if pipe.do_classifier_free_guidance:
                    prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
                    pooled_prompt_embeds = torch.cat([negative_pooled_prompt_embeds, pooled_prompt_embeds], dim=0)

                callback_kwargs["prompt_embeds"] = prompt_embeds
            return callback_kwargs

        if switch_time == 0:
            image = pipeline(
                prompt=prompts[1],
                num_inference_steps=28,
                guidance_scale=4.5,
                max_sequence_length=512,
                generator=torch.Generator(device="cpu").manual_seed(seed),
            ).images[0]
        else:
            image = pipeline(
                prompt=prompts[0],
                num_inference_steps=28,
                guidance_scale=4.5,
                max_sequence_length=512,
                generator=torch.Generator(device="cpu").manual_seed(seed),
                callback_on_step_end=decode_tensors,
                callback_on_step_end_tensor_inputs=["latents", "prompt_embeds"],
            ).images[0]

        image.save(f"{out_dir}/re_{switch_time}.jpeg")

class update_prompt():
    def __init__(self, results_path, currtime, experiment_name, seed, cache_path=None):
        self.experiment_name = experiment_name
        self.seed = seed
        self.contents = read_jsonl(results_path)
        self.out_dir = Path(results_path).parent
        self.currtime = currtime
        self.cache_path = cache_path
    def process_one_by_one(self, start_id=0):
        self.by_batch = False
        self.cache_path = f'{self.out_dir}/response_{self.seed}_{self.experiment_name}_{self.currtime}.jsonl'
        client = OpenAI()
        out = []
        print(f'Processing one by one!!.')
        for i, content in enumerate(self.contents[start_id:]):
            image_in_message = True
            if "prompt" in content.keys():
                original_prompt = content["prompt"]
            else:
                original_prompt = content["original_prompt"]
            if "detailscribe" in self.experiment_name:
                topic = content["topic"].split('_')[0]
                text_prompt = (
                    f'This is an image generated with the prompt: "{original_prompt}" '
                    f'But this image looks bizarre. '
                    f'Examine the image carefully follow the concept of "{topic}"'
                    f' attached below and other components in the image. For each abnormal part, '
                    f'describe what is wrong with it, then give a concise description on how to correct it.\n'
                    f'Components: {content["components"]}\n'
                    f'Do not simply rely on the components described above, but also exam whether an object looks complete.\n'
                    f'First, write your answer in a numbered list,\n'
                    f'Then, rank the issues by their degree of impact on presenting the concept.\n'
                    f'Last, summarize the correction instructions in order, and write a new description with the first sentence to be "{original_prompt}" followed by correction instructions.\n'
                    f'Do not change the first sentence. '
                    f'Be concise, no more than 70 words, but make sure not to miss any information needs to be corrected. '
                    f'Provide the new description in angle brackets <>.\n'
                    f'The components described in the original prompt are essential, '
                    f'do not question the concepts in the original prompt.')
            elif "gpt_refine" in self.experiment_name.lower():
                text_prompt = (f'Based on the original intent to depict "{original_prompt}", '
                               f'suggest a concise prompt (no more than 70 words)'
                               f' for refining this image:')
            elif "abl" in self.experiment_name:
                text_prompt = (
                    f'This is an image generated with the prompt: "{original_prompt}" '
                    f'But this image looks bizarre. '
                    f'Examine the image carefully. For each abnormal part, '
                    f'describe what is wrong with it, then give a concise description on how to correct it.\n'
                    f'First, write your answer in a numbered list,\n'
                    f'Then, summarize the correction instructions, and write a new description with the first sentence to be "{original_prompt}" followed by correction instructions.\n'
                    f'Do not change the first sentence. '
                    f'Be concise, no more than 70 words, but make sure not to miss any information needs to be corrected. '
                    f'Provide the new description in angle brackets <>.\n'
                    f'The components described in the original prompt are essential, '
                    f'do not question the concepts in the original prompt.')
            elif "gpt_rewrite" in self.experiment_name:
                image_in_message = False
                text_prompt = (
                    f'Given text prompt: "{original_prompt}"\n'
                    f'Can you come up with a detailed description of no more than 70 words for this prompt? '
                    f'Provide the new description in angle brackets <>.')

            if image_in_message:
                base64_image = encode_image(f'{content["dir"]}/original_{self.seed}.jpeg')
                messages = [
                               {"type": "image_url",
                                "image_url":
                                    {"url": f"data:image/jpeg;base64,{base64_image}"}},
                               {"type": "text", "text": text_prompt},
                           ]
            else:
                messages = [{"type": "text", "text": text_prompt}, ]

            completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {
                        "role": "user",
                        "content": messages
                    }
                ]
            )
            msg = completion.choices[0].message
            content["custom_id"] = f"request-{i:04}"
            content["response"] = msg.content
            out.append(content)
            with open(self.cache_path, 'a') as f:
                f.write(json.dumps(content) + '\n')
        print(f'evaluation finished. save cache in {self.cache_path}')
        return self.cache_path

    def process_batch(self, cached_path=None):
        cached_path = self.cache_path if cached_path is None else cached_path
        if cached_path is None:
            print(f'Error: Cannot fetch response')
            return
        self.cache_path = cached_path
        with open(cached_path, 'r') as f:
            lines = [json.loads(line) for line in f]
        out = []
        for i, each in enumerate(lines):
            if each == {}:
                continue
            custom_id = int(each["custom_id"].split('-')[-1])
            if self.by_batch:
                answer = each["response"]["body"]["choices"][0]["message"]["content"]
            else:
                answer = each["response"]
            if "gpt_refine" in self.experiment_name:
                new_prompt = answer
            else:
                text = re.findall(r"<(.*?)>", answer)
                if len(text) == 0:
                    continue
                new_prompt = text[-1]
            if "prompt" in self.contents[custom_id].keys():
                original_prompt = self.contents[custom_id]["prompt"]
            else:
                original_prompt = self.contents[custom_id]["original_prompt"]
            data = {
                "dir": self.contents[custom_id]["dir"],
                "topic": self.contents[custom_id]["topic"],
                "original_prompt": original_prompt,
                "new_prompt": new_prompt
            }
            out.append(data)

        # self.new_prompt_path = self.cache_path.replace('response', 'new_prompt')
        self.new_prompt_path = f'{self.out_dir}/new_prompt_{self.seed}_{self.experiment_name}.jsonl'
        write_jsonl(out, self.new_prompt_path)
        print(f'new prompts saved at {self.new_prompt_path}')
        return self.new_prompt_path


def run_baseline(pipeline, out_dir, data, seed, dalle3=False):
    if not dalle3:
        start_time = time.time()
        image = pipeline(
            prompt=data['new_prompt'],
            num_inference_steps=28,
            guidance_scale=4.5,
            max_sequence_length=512,
            generator=torch.Generator(device="cpu").manual_seed(seed)
        ).images[0]
        each_time = time.time() - start_time
        out_path = f'{out_dir}/GPT_rewrite_{seed}.jpeg'
        image.save(out_path)
        print(f'image saved at {out_path}')
    else:
        # for DaLLE3
        client = OpenAI()
        start_time = time.time()
        try:
            response = client.images.generate(
                model="dall-e-3",
                prompt=data['original_prompt'],
                size="1024x1024",
                quality="standard",
                n=1,
            )
        except Exception as e:
            print(e)
            print(f'Dalle3 failed at {data["topic"]}')
            return
        each_time = time.time() - start_time
        dalle3_filename = f'{out_dir}/dalle3.jpeg'
        image_url = response.data[0].url
        img = requests.get(image_url, timeout=5)
        if img.status_code == 200:
            with open(dalle3_filename, 'wb') as f:
                f.write(img.content)
                print(f'image saved at {dalle3_filename}')
        else:
            print(response.data[0])
    return each_time


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_path", type=str,
                        default=None)  # default="test_data/abstract_concepts/concepts.jsonl")
    parser.add_argument("--seed", type=int, default=2628670643)
    parser.add_argument("--catlog_path", type=str, default=None)
    parser.add_argument("--cached_path", type=str, default=None)
    parser.add_argument("--new_prompt_path", type=str, default=None)
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="detailscribe",
    )
    parser.add_argument("--out_dir", type=str, default="results")
    args = parser.parse_args()

    # init model sd3.5
    model_id = "stabilityai/stable-diffusion-3.5-large"
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model_nf4 = SD3Transformer2DModel.from_pretrained(
        model_id,
        subfolder="transformer",
        quantization_config=nf4_config,
        torch_dtype=torch.bfloat16
    )
    pipeline = StableDiffusion3Pipeline.from_pretrained(
        model_id,
        transformer=model_nf4,
        torch_dtype=torch.bfloat16
    )
    pipeline.enable_model_cpu_offload()
    currtime = time.strftime("%m%d%H%M")

    catlog_path = args.catlog_path # for init generation
    cached_path = args.cached_path
    new_prompt_path = args.new_prompt_path
    experiment_name = args.experiment_name.lower()
    assert experiment_name in ['init_only', 'gpt_refine', 'detailscribe', 'abl', 'gpt_rewrite', 'dalle3']

    if args.prompt_path is not None:
        out_dir = f'{args.out_dir}/{args.prompt_path.split("/")[-1].split(".")[0]}'
        Path(out_dir).mkdir(parents=True, exist_ok=True)
    else:
        input_path = next((s for s in [new_prompt_path, cached_path, catlog_path] if s is not None), None)
        out_dir = str(Path(input_path).parent)

    init_gen_time = 0
    critique_time = 0
    refinement_time = 0

    if 'dalle3' not in experiment_name:
        if new_prompt_path is None:
            if cached_path is None:
                if catlog_path is None:
                    if 'gpt_rewrite' in experiment_name:
                        catlog_path = args.prompt_path
                    else:
                        """
                        experiment_name's needs original image:
                        init_only, gpt_refine, detailscribe, abl
                        Read in prompts and generate original image one by one, and create a catlog
                        """
                        if args.prompt_path is None:
                            print('Error: please at least one of prompt or catlog path')
                            exit()
                        prompts = read_jsonl(args.prompt_path)
                        catlogs = []
                        start_init_gen = time.time()
                        for prompt in prompts:
                            print(f'generating according to prompt:\n{prompt}')
                            topic_dir = run_original(pipeline, prompt, out_dir, seed=args.seed)
                            prompt["dir"] = topic_dir
                            catlogs.append(prompt)
                        catlog_path = f"{out_dir}/catlogs_{args.seed}.jsonl"
                        write_jsonl(catlogs, catlog_path)
                        end_init_gen = time.time()
                        init_gen_time = end_init_gen - start_init_gen

                print(f'reading init gen from {catlog_path}')
                if experiment_name == "init_only":
                    exit()
                    
                start_critique = time.time()
                prompt_updater = update_prompt(catlog_path, currtime, experiment_name, args.seed)
                cached_path = prompt_updater.process_one_by_one()
                end_critique = time.time()
                critique_time = end_critique - start_critique
            else:
                prompt_updater = update_prompt(catlog_path, currtime, experiment_name, cached_path)
        # fetch new prompt from cache
            new_prompt_path = prompt_updater.process_batch(cached_path)
    else:
        new_prompt_path = next((s for s in [new_prompt_path, cached_path, catlog_path, args.prompt_path] if s is not None), None)

    new_prompts = read_jsonl(new_prompt_path)
    logs = []
    start_refinement = time.time()
    baseline_time = 0
    if any(experiment in experiment_name for experiment in ['gpt_rewrite', 'gpt_refine']):
        redenoise_steps = [0]
    else:
        redenoise_steps = [0, 1, 2, 4]
    # update_dir = False
    for i, each in enumerate(new_prompts):
        print(f'generating {i+1}/{len(new_prompts)}.')
        print(f'generating according to data:\n{each}')
        if each['dir'] is None:
            # update_dir = True
            each['dir'] = f'{out_dir}/{each["topic"]}'
        if any(experiment in experiment_name for experiment in ['gpt_rewrite', 'dalle3']):
            baseline_time += run_baseline(pipeline, each['dir'], each, args.seed, dalle3='dalle3' in experiment_name)
        else:
            inference_pos(pipeline,
                          [each["original_prompt"], each["new_prompt"]],
                          each['dir'],
                          experiment_name,
                          max_distance= redenoise_steps,
                          seed=args.seed,
                          step_size=1)
            logs.append(f'generated for {each["topic"]}, saved in {each["dir"]}')
    end_refinement = time.time()
    refinement_time = end_refinement - start_refinement
    with open(f"{out_dir}/logs_{currtime}.txt", 'w') as f:
        for each in logs:
            f.write(each + '\n')

    print(f'init_gen_time: {init_gen_time}, critique_time: {critique_time}, re-denoise_time: {refinement_time}')
