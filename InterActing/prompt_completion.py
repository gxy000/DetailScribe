import copy
import os.path
from collections import defaultdict
from importlib.resources import contents
from turtledemo.penrose import start

import pandas as pd

from diffusers import BitsAndBytesConfig, SD3Transformer2DModel
from diffusers import StableDiffusion3Pipeline
import torch
from pathlib import Path
import json
import time
import base64
import textwrap
import re
import jsonlines
import math
from openai import OpenAI
import argparse
import openai
import random

def write_jsonl(contents, path):
    with open(path, 'w') as f:
        for each in contents:
            f.write(json.dumps(each) + '\n')

class prompt_generator():
    def __init__(self, data_dir, scenario, currtime, prompt_type, cache_path=None, cache_paths=None):
        # read text file
        self.type = prompt_type
        self.scenario = f'{scenario}'

        if self.type == "original_prompt":
            sample_path_txt = os.path.join(data_dir, f'relations_txt/{scenario}.txt')
            print(f'reading from {sample_path_txt}')
            with open(sample_path_txt, 'r') as f:
                contents_list = f.readlines()[0].split(',')
                if "\'" in contents_list[0]:
                    self.contents = [re.findall(r"'(.*?)'", content)[0] for content in contents_list]
                else:
                    self.contents = contents_list
        elif self.type == "decomposition":
            self.contents = []
            sample_path_jsonl = os.path.join(data_dir, f'original_prompt/{scenario}.jsonl')
            with jsonlines.open(sample_path_jsonl) as reader:
                self.contents = [each for each in reader]
        self.out_dir = os.path.join(data_dir, f'{self.type}')
        Path(self.out_dir).mkdir(parents=True, exist_ok=True)
        self.out_path = os.path.join(self.out_dir, f'{scenario}.jsonl')

        self.currtime = currtime

        if cache_paths is None:
            self.cache_paths = []
        else:
            self.cache_paths = cache_paths
        self.cache_path = cache_path

    def process_one_by_one(self, start_idx=0):
        self.cache_path = f'{self.out_dir}/response_{self.currtime}.jsonl'
        client = OpenAI()
        out = []
        print(f'process one by one.')

        for i, content in enumerate(self.contents):
            if i >= start_idx:
                if self.type == "original_prompt":
                    if "abstract_concept" in self.scenario:
                        text_prompt = (
                            f'come up with a description of a scene of "{content}" of novel combination, '
                            f'the description should be similar as the following example and be uncommon being observed. '
                            f'Do not use passive voice.\n'
                            f'Double check the description to focus on major relation, which is {content}.\n'
                            f'Write down your answer in this format: '
                            f'{{"topic": {content}, "prompt": description}}\n'
                            f'For example:\n'
                            f'concept: "tic−tac−toe"\n'
                            f'Description: A tic−tac−toe composed by tomato and cucumber as the players symbols.\n'
                            f'Then, the output should be: '
                            f'{{"topic": "tic−tac−toe", "prompt": "A tic−tac−toe composed by tomato and cucumber as the players symbols."}}')
                    elif "multi_subject" in self.scenario:
                        text_prompt = (
                            f'come up with a description of two animals doing "{content}", '
                            f'the description should be similar as the following example and be uncommon being observed. '
                            f'Do not use passive voice.\n'
                            f'Double check the description to focus on major relation, which is {content}.\n'
                            f'Write down your answer in this format: '
                            f'{{"topic": {content}, "prompt": description}}\n'
                            f'The description must contains the exact "{content}" word.\n'
                            f'For example:\n'
                            f'concept: "High-Fiving"\n'
                            f'Description: A dolphin and a seal leap from the water, high-fiving with their flippers.\n'
                            f'Then, the output should be: '
                            f'{{"topic": "High-Fiving", "prompt": "A dolphin and a seal leap from the water, high-fiving with their flippers."}}')
                    else:
                        text_prompt = (
                            f'come up with a description of an animal "{content}", '
                            f'the description should be similar as the following example and be uncommon being observed. '
                            f'Do not use passive voice.\n'
                            f'Double check the description to focus on major relation, which is {content}.\n'
                            f'Write down your answer in this format: '
                            f'{{"topic": {content}, "prompt": description}}\n'
                            f'For example:\n'
                            f'interaction: "taking photos"\n'
                            f'Entities: squirrel\n'
                            f'Description: A squirrel taking photos with a camera.\n'
                            f'Then, the output should be: '
                            f'{{"topic": "taking-photos", "prompt": "A squirrel taking photos with a camera."}}')
                elif self.type == "decomposition":
                    text_prompt = (f'For each abstract concept, '
                                   f'we can decompose it into interactions defined by contact points and contact objects.\n'
                                   f'For example, (concept: cooking)= (hand hold the handle of a ladle) + (ladle stir the ingredient in the pot) + (pot is on a stove)\n'
                                   f'Please decompose the concept "{content["topic"]}" based on the prompt  "{content["prompt"]}" '
                                   f'in the same format without explanation.\n'
                                   f'Keep the program simple.\n'
                                   f'Use only the most necessary parts of the schema that '
                                   f'can be mapped to objects in an image.\n'
                                   f'Describe only the interactions that can happen simultaneously.\n'
                                   f'Provide your answer in angle brackets <>.\n'
                                   f'For example:\n'
                                   f'When you try to decompose the concept "squirrel-carve-acorn" based on the prompt '
                                   f'"An anime of a squirrel carving a design on an acorn using a chisel."\n'
                                   f'Your answer should be: <(concept: squirrel-carve-acorn) = (paw hold acorn) + (chisel carve design) + (acorn rest on ground)>.')
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
                # if "content" in msg.keys():
                if type(content) == str:
                    line = {"content": msg.content}
                else:
                    line = content
                    line["content"] = msg.content

                out.append(line)
        write_jsonl(out, self.cache_path)
        print(f'evaluation finished. save cache in {self.cache_path}')
        self.cache_paths.append(self.cache_path)
        return self.cache_path
    def process_batch(self, cached_paths=None):
        out = []
        cached_paths = self.cache_paths if cached_paths is None else cached_paths
        if cached_paths is None:
            if cached_paths is None:
                print(f'Error: Cannot fetch response')
                return
        lines = []
        print(f'{cached_paths=}')
        for cached_path in cached_paths:
            with open(cached_path, 'r') as f:
                lines = lines + [json.loads(line) for line in f]

        for i, each in enumerate(lines):
            if each == {}:
                continue
            answer = each["content"]

            try:
                if self.type == "original_prompt":
                    if self.scenario in ["abstract_concept", "multi_subjects"]:
                        strings = re.findall(r"<.*?>", answer)
                    else:
                        strings = re.findall(r"\{.*?\}", answer)
                elif self.type == "decomposition":
                    strings = re.findall(r"<.*?>", answer)

                if len(strings) > 0:
                    comp_string = strings[-1].replace("<", "").replace(">", "")
                else:
                    continue

                try:
                    if self.type == "original_prompt":
                        if self.scenario in ["abstract_concept", "multi_subjects"]:
                            if len(strings) < 2:
                                continue
                            prompt = {"topic": strings[0].replace("Concept: ", "").replace("<", "").replace(">", ""),
                                      "prompt": strings[1].replace("Image description: ", "").replace("<", "").replace(">", "")}
                        else:
                            prompt = json.loads(comp_string)
                        prompt["topic"] = prompt["topic"].replace(" ", "-")
                        out.append(prompt)
                    elif self.type == "decomposition":
                        out.append({"topic": each["topic"], "components": comp_string, "prompt": each["prompt"]})
                except json.JSONDecodeError as e:
                    print(f'Error at string: {comp_string}')
                    print(f"JSONDecodeError: {e}")
            except json.JSONDecodeError as e:
                print(f'Error at string: {answer}')
                print(f"JSONDecodeError: {e}")

        write_jsonl(out, self.out_path)
        print(f'new prompts saved at {self.out_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str,
                        default="InterActing/data/")
    parser.add_argument("--scenario", type=str,
                        default="rare_relations_top100")
    # cache_path=None, cache_paths=None, batch_size=50
    parser.add_argument("--cached_path", type=str,
                        default=None)
    parser.add_argument("--cached_paths", nargs='+' , type=str,
                        default=[])
    parser.add_argument("--type", type=str,
                        default="original_prompt")

    args = parser.parse_args()

    currtime = time.strftime("%m%d%H%M")

    generator = prompt_generator(args.data_dir, args.scenario, currtime, args.type, args.cached_path, args.cached_paths)

    if len(args.cached_paths) > 0:
        generator.cache_paths = args.cached_paths

    if args.cached_path is None:
        generator.process_one_by_one()
    else:
        generator.cache_paths = [args.cached_path]
    generator.process_batch()
