import argparse
import clip
import torch
from PIL import Image
from sklearn.preprocessing import normalize
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import torch
import tqdm
import numpy as np
import sklearn.preprocessing
import collections
import os
import pathlib
import json
import jsonlines
import pprint
import warnings
from packaging import version
import pandas as pd
import random
from pathlib import Path

import ImageReward as RM
from transformers import AutoProcessor, AutoModel, CLIPFeatureExtractor, CLIPImageProcessor, AutoTokenizer

import collections
import time
import base64
import textwrap
import re
import math
from openai import OpenAI
import spacy
import sys
from tqdm import tqdm, trange

BLIPvqa_path = '/home/gxy/T2I-CompBench/BLIPvqa_eval'
sys.path.append(BLIPvqa_path)
from BLIP.train_vqa_func import VQA_main

GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'eval_json',
        nargs='+' ,
        type=str,
        help='Candidates json mapping from image_id --> candidate.')

    parser.add_argument('--ablation', action='store_true', help='if set, we will evaluate ablation')

    parser.add_argument(
        '--references_json',
        default=None,
        help='Optional references json mapping from image_id --> [list of references]')

    parser.add_argument(
        '--compute_other_ref_metrics',
        default=1,
        type=int,
        help='If references is specified, should we compute standard reference-based metrics?')

    parser.add_argument(
        '--save_per_instance',
        default=None,
        help='if set, we will save per instance clipscores to this file')

    parser.add_argument(
        '--metric',
        default='CLIPScore',
        help='Which metric to compute. Options are CLIPScore, ImageReward')

    parser.add_argument(
        '--by_scale',
        action="store_true",
    )

    args = parser.parse_args()

    if isinstance(args.save_per_instance, str) and not args.save_per_instance.endswith('.json'):
        print('if you\'re saving per-instance, please make sure the filepath ends in json.')
        quit()
    return args

currtime = time.strftime("%m%d%H%M")

"""
VLM Evaluation: gpt-4o or gemini-2.0-flash
"""
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


def api_process_one_by_one(topic, prompt, image_path, client, response_path, model="gpt-4o", scale=False, spatial=False):
    if scale:
        if spatial:
            text_prompt = (f"You are my assistant to identify objects and their spatial layout in the image. \
                According to the image, evaluate if the text \"{prompt}\" is correctly portrayed in the image. \
                Give a score from 1 to 5 according the criteria: \n\
                    5: correct spatial layout ({topic}) in the image for all objects mentioned in the text. \
                    4: basically, spatial layout of objects matches the text. \
                    3: spatial layout not aligned properly with the text. \
                    2: image not aligned properly with the text. \
                    1: image almost irrelevant to the text. \n \
                    Return the score in angle brackets <>. For example, if the image's spatial layout of objects matches the text and got score 4, response: <4>")
        else:
            text_prompt = (
                f'The above images were generated with the prompt: "{prompt}" '
                f'Please rate text-image alignment score of each image from 1 to 5, '
                f'focus on {topic} and follow the criteria:\n'
                f'1: poor interaction, subject(s) not acting correctly,\n'
                f'2: subject(s) incorrect/inaccurate\n'
                f'3: critical part missing (e.g. missing critical tools or patterns to complete {topic}),\n'
                f'4: nearly perfect but some subparts need further improvement (e.g. needs to refine appearance of tools or limbs, '
                f'subject is not {topic} correctly),\n'
                f'5: image perfectly depicts {topic} .\n'
                f'Return the score in angle brackets <>.'
                f'For example, if the image is nearly perfect and got score 4, response: <4>'
            )
    else:
        text_prompt = (f'The above images were generated with the prompt: "{prompt}" '
                       f'When satisfies both of 1. the number of objects in image matches the prompt, '
                       f'2.image perfectly depict the relation of "{topic}" '
                       f'between the subjects, response: <yes>, otherwise <no>')

    base64_image = encode_image(image_path)
    messages = [{"type": "image_url",
                 "image_url":
                     {"url": f"data:image/jpeg;base64,{base64_image}"}
                 }, ]
    messages.append({"type": "text", "text": text_prompt})
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": messages
            }
        ]
    )
    msg = completion.choices[0].message
    response = msg.content
    print(f'{topic} response: {response}')
    with open(response_path, 'a') as f:
        f.write(json.dumps(response) + '\n')
    if scale:
        ss = re.findall(r"<(.*?)>", response)
        score = 2
        if len(ss) > 0:
            if ss[0].isdigit():
                score = int(ss[0])
    else:
        score = 1 if 'yes' in response else 0
    return score, response

'''
Code for CLIPScore (https://arxiv.org/abs/2104.08718)
@inproceedings{hessel2021clipscore,
  title={{CLIPScore:} A Reference-free Evaluation Metric for Image Captioning},
  author={Hessel, Jack and Holtzman, Ari and Forbes, Maxwell and Bras, Ronan Le and Choi, Yejin},
  booktitle={EMNLP},
  year={2021}
}
'''
class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data, prefix='A photo depicts'):
        self.data = data
        self.prefix = prefix
        if self.prefix[-1] != ' ':
            self.prefix += ' '

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)

class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # only 224x224 ViT-B/32 supported for now
        self.preprocess = self._transform_test(224)

    def _transform_test(self, n_px):
        return Compose([
            Resize(n_px, interpolation=Image.BICUBIC),
            CenterCrop(n_px),
            lambda image: image.convert("RGB"),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_captions(captions, model, device, batch_size=256, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPCapDataset(captions),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, device, batch_size=64, num_workers=8):
    data = torch.utils.data.DataLoader(
        CLIPImageDataset(images),
        batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in tqdm(data):
            b = b['image'].to(device)
            if device == 'cuda':
                b = b.to(torch.float16)
            all_image_features.append(model.encode_image(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def get_clip_score(model, images, candidates, device, w=2.5):
    '''
    get standard image-text clipscore.
    images can either be:
    - a list of strings specifying filepaths for images
    - a precomputed, ordered matrix of image features
    '''
    if isinstance(images, list):
        # need to extract image features
        images = extract_all_images(images, model, device)

    candidates = extract_all_captions(candidates, model, device)

    # as of numpy 1.21, normalize doesn't work properly for float16
    if version.parse(np.__version__) < version.parse('1.21'):
        images = sklearn.preprocessing.normalize(images, axis=1)
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')
        images = images / np.sqrt(np.sum(images ** 2, axis=1, keepdims=True))
        candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))

    per = w * np.clip(np.sum(images * candidates, axis=1), 0, None)
    return np.mean(per), per, candidates


def get_refonlyclipscore(model, references, candidates, device):
    '''
    The text only side for refclipscore
    '''
    if isinstance(candidates, list):
        candidates = extract_all_captions(candidates, model, device)

    flattened_refs = []
    flattened_refs_idxs = []
    for idx, refs in enumerate(references):
        flattened_refs.extend(refs)
        flattened_refs_idxs.extend([idx for _ in refs])

    flattened_refs = extract_all_captions(flattened_refs, model, device)

    if version.parse(np.__version__) < version.parse('1.21'):
        candidates = sklearn.preprocessing.normalize(candidates, axis=1)
        flattened_refs = sklearn.preprocessing.normalize(flattened_refs, axis=1)
    else:
        warnings.warn(
            'due to a numerical instability, new numpy normalization is slightly different than paper results. '
            'to exactly replicate paper results, please use numpy version less than 1.21, e.g., 1.20.3.')

        candidates = candidates / np.sqrt(np.sum(candidates ** 2, axis=1, keepdims=True))
        flattened_refs = flattened_refs / np.sqrt(np.sum(flattened_refs ** 2, axis=1, keepdims=True))

    cand_idx2refs = collections.defaultdict(list)
    for ref_feats, cand_idx in zip(flattened_refs, flattened_refs_idxs):
        cand_idx2refs[cand_idx].append(ref_feats)

    assert len(cand_idx2refs) == len(candidates)

    cand_idx2refs = {k: np.vstack(v) for k, v in cand_idx2refs.items()}

    per = []
    for c_idx, cand in tqdm(enumerate(candidates)):
        cur_refs = cand_idx2refs[c_idx]
        all_sims = cand.dot(cur_refs.transpose())
        per.append(np.max(all_sims))

    return np.mean(per), per


def eval_by_cscore(image_ids, image_paths, candidates, device, references_json=None):
    model, transform = clip.load("ViT-B/32", device=device, jit=False,
                                 download_root='../3rd_party/inputs/clip')
    model.eval()

    image_feats = extract_all_images(
        image_paths, model, device, batch_size=64, num_workers=8)

    # get image-text clipscore
    _, per_instance_image_text, candidate_feats = get_clip_score(
        model, image_feats, candidates, device)

    if references_json:
        with open(references_json) as f:
            references = json.load(f)
            references = [references[cid] for cid in image_ids]
            if isinstance(references[0], str):
                references = [[r] for r in references]
        _, per_instance_text_text = get_refonlyclipscore(
            model, references, candidate_feats, device)
        # F-score
        refclipscores = 2 * per_instance_image_text * per_instance_text_text / (
                per_instance_image_text + per_instance_text_text)
        scores = {image_id: {'CLIPScore': float(clipscore), 'RefCLIPScore': float(refclipscore)}
                  for image_id, clipscore, refclipscore in
                  zip(image_ids, per_instance_image_text, refclipscores)}
    else:
        scores = {image_id: float(clipscore)
                  for image_id, clipscore in
                  zip(image_ids, per_instance_image_text)}
    return scores


'''
Code for BLIPvqa_eval (https://github.com/Karine-Huang/T2I-CompBench/tree/main/BLIPvqa_eval)
@article{huang2023t2icompbench,
  title={T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation},
  author={Huang, Kaiyi and Sun, Kaiyue and Xie, Enze and Li, Zhenguo and Liu, Xihui},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  pages={78723--78747},
  year={2023}
}
'''

def Create_annotation_for_BLIP(caption_list, image_path_list, outdir, np_index=None):
    nlp = spacy.load("en_core_web_sm")

    annotations = []
    cnt = 0
    for caption, image in zip(caption_list, image_path_list):
        image_dict = {}
        image_dict['image'] = image
        image_dict['question_id'] = cnt
        doc = nlp(caption)

        noun_phrases = []
        for chunk in doc.noun_chunks:
            if chunk.text not in ['top', 'the side', 'the left', 'the right']:  # todo remove some phrases
                noun_phrases.append(chunk.text)
        if (len(noun_phrases) > np_index):
            q_tmp = noun_phrases[np_index]
            image_dict['question'] = f'{q_tmp}?'
        else:
            image_dict['question'] = ''

        image_dict['dataset'] = "color"
        cnt += 1

        annotations.append(image_dict)

    print('Number of Processed Images:', len(annotations))

    json_file = json.dumps(annotations)
    with open(f'{outdir}/vqa_test.json', 'w') as f:
        f.write(json_file)


def stats(topics, all_scores, abl_flag, log_path = None, multi_seed = None, captions=None, stats_dir="results/eval/stats"):

    if abl_flag:
        groups = ["abl_0", "abl_1", "abl_2", "abl_4",
                  "0", "1", "2", "4"]
        methods = ["abl", "DetailScribe"]
    else:
        groups = ["original", "gpt_rewrite", "dalle3", "multi_seed", "gpt_refine", "0", "1", "2", "4"]
        methods = ["SD", "+GPT Rewrite", "DALLE3", "+multi-seed", "+GPT Refine", "DetailScribe"]

    all_ratings_df = []
    for i, topic in enumerate(topics):
        scores = []
        caption = None
        for j, group in enumerate(groups):
            image_id = f'{topic}_{group}'
            caption = captions[image_id] if captions else None
            scores.append(all_scores[image_id])
        if multi_seed:
            new_scores = [*scores, max(scores)]
        else:
            if abl_flag:
                new_scores = [max(scores[:4]), max(scores[4:])]
            else:
                new_scores = [*scores[:-4], max(scores[-4:])]
        for j, ss in enumerate(new_scores):
            all_ratings_df.append({
                "example_id": i,
                "topic": topic,
                "caption": caption,
                "method": methods[j],
                "score": ss,
                "is_win": ss == np.max(new_scores)})

    all_ratings_df = pd.DataFrame(all_ratings_df)
    result = all_ratings_df.groupby("method").aggregate({
        "score": ["mean", "std", "sem"],
    })
    print("\n\n=== Method Scores Mean Std")
    print(result)

    print("\n\n=== Win Rate Mean Std")
    print(all_ratings_df.groupby(["method"]).aggregate({
        "is_win": "mean"
    }))

    if log_path is not None:
        if abl_flag:
            log_path = log_path.replace('.json', '_abl.json')
        print(f'writing to {log_path}')
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        all_ratings_df.to_json(log_path, orient="records", lines=True)

    Path(stats_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{stats_dir}/scores.txt", "a") as file:
        # Use print with the file parameter
        print(f'at {time.strftime("%m%d%H%M")}', file=file)
        print(f'{log_path}', file=file)
        print("=== Method Scores Mean Std", file=file)
        print(result, file=file)
        print(all_ratings_df.groupby(["method"]).aggregate({
            "is_win": "mean"
        }), file=file)
        print("\n\n\n", file=file)


def main():
    args = parse_args()

    image_paths, topics, image_ids = [], [], []
    captions = {}
    image_path_dict = {}
    names_by_topic = collections.defaultdict(list)
    for i, path in enumerate(args.eval_json):
        with jsonlines.open(path) as reader:
            for each in reader:
                topic = each['topic']
                topics.append(topic)
                for step, path in zip(each['steps'], each['image_paths']):
                    image_name = f'{topic}_{step}'

                    image_paths.append(path)
                    image_ids.append(image_name)
                    image_path_dict[image_name] = path
                    names_by_topic[topic].append(image_name)
                    captions[image_name] = each['original_prompt']

    print(f'Read in {len(topics)}, len of image_paths: {len(image_paths)}')

    candidates = captions
    candidates = [candidates[cid] for cid in image_ids]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == 'cpu':
        warnings.warn(
            'CLIP runs in full float32 on CPU. Results in paper were computed on GPU, which uses float16. '
            'If you\'re reporting results on CPU, please note this when you report.')

    if args.metric == "CLIPScore":
        scores = eval_by_cscore(image_ids, image_paths, candidates, device, args.references_json)
        print('CLIPScore: {:.4f}'.format(np.mean([s for s in scores.values()])))
    elif args.metric == "ImageReward":
        model = RM.load("ImageReward-v1.0")
        scores = {}
        for caption, image_id, image_path in zip(candidates, image_ids, image_paths):
            scores[image_id] = model.score(caption, image_path)
    elif args.metric == "gpt":
        spatial = True if "text_spatial_rel" in args.eval_json[0] else False
        scores = {}
        responses = []
        client = OpenAI()
        print(f'all topics: {topics}')
        response_path = f'Evaluation/evaluation_log/responses_{args.metric}_{currtime}.jsonl'

        with open(response_path, 'w') as f:
            f.write("")
        for caption, image_id, image_path in zip(candidates, image_ids, image_paths):
            scores[image_id], response = api_process_one_by_one(image_id.split("_")[0], caption, image_path, client, response_path, model="gpt-4o", scale=args.by_scale, spatial=spatial)
            responses.append(response)
    elif args.metric == "gemini":
        spatial = True if "text_spatial_rel" in args.eval_json[0] else False
        scores = {}
        responses = []
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
        response_path = f'Evaluation/evaluation_log/responses_{args.metric}_{currtime}.jsonl'
        with open(response_path, 'w') as f:
            f.write("")
        for i, (caption, image_id, image_path) in enumerate(zip(candidates, image_ids, image_paths)):
            scores[image_id], response = api_process_one_by_one(image_id.split("_")[0], caption, image_path, client, response_path, model="gemini-2.0-flash", scale=args.by_scale, spatial=spatial)
            responses.append(response)
            if i%15 == 0:
                time.sleep(60)
    elif args.metric == "BLIP-VQA":
        np_index = 8  # how many noun phrases

        answer = []
        sample_num = len(candidates)
        reward = torch.zeros((sample_num, np_index)).to(device='cuda')
        out_dir = f'Evaluation/evaluation_log/VQA_{args.metric}_{currtime}'
        os.makedirs(out_dir, exist_ok=True)

        order = "_blip"  # rename file
        for i in tqdm(range(np_index)):
            print(f"start VQA{i + 1}/{np_index}!")
            os.makedirs(f"{out_dir}/annotation{i + 1}{order}", exist_ok=True)
            os.makedirs(f"{out_dir}/annotation{i + 1}{order}/VQA/", exist_ok=True)
            Create_annotation_for_BLIP(
                candidates, image_paths,
                f"{out_dir}/annotation{i + 1}{order}",
                np_index=i,
            )
            answer_tmp = VQA_main(f"{out_dir}/annotation{i + 1}{order}/",
                                  f"{out_dir}/annotation{i + 1}{order}/VQA/")
            answer.append(answer_tmp)

            with open(f"{out_dir}/annotation{i + 1}{order}/VQA/result/vqa_result.json", "r") as file:
                r = json.load(file)
            with open(f"{out_dir}/annotation{i + 1}{order}/vqa_test.json", "r") as file:
                r_tmp = json.load(file)
            for k in range(len(r)):
                if (r_tmp[k]['question'] != ''):
                    reward[k][i] = float(r[k]["answer"])
                else:
                    reward[k][i] = 1
            print(f"end VQA{i + 1}/{np_index}!")
        reward_final = reward[:, 0]
        for i in range(1, np_index):
            reward_final *= reward[:, i]

        scores = {image_id: reward_final
                  for image_id, reward_final in
                  zip(image_ids, list(reward_final.cpu().numpy()))}

    print(f'stats for {args.eval_json}')
    stats_path = f'results/eval/stats/{args.eval_json[0].split("/")[-1].split(".")[0]}/{args.metric}.jsonl'
    if args.metric in ["gpt", "gemini"]:
        if not args.by_scale:
            stats_path = stats_path.replace('.jsonl', 'binary.jsonl')

    stats(topics, scores, args.ablation, stats_path, captions=captions)

    if args.save_per_instance:
        with open(args.save_per_instance, 'w') as f:
            f.write(json.dumps(scores))


if __name__ == '__main__':
    main()



