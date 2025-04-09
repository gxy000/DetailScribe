import jsonlines
import json
import argparse
import collections

def write_jsonl(contents, path):
    with open(path, 'w') as f:
        for each in contents:
            f.write(json.dumps(each) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='data.jsonl')
    parser.add_argument('--type', type=str, default='original_prompt')
    args = parser.parse_args()


    topics = collections.defaultdict(int)
    with jsonlines.open(args.input) as reader:
        out = []
        for each in reader:
            topic = each['topic']
            if args.type == 'original_prompt':
                if not (each['topic'].lower().replace('-', ' ') in each['prompt'].lower().replace('-', ' ') and len(each['topic']) > 1):
                    print(f'bad line: {each}')
                    continue
            topics[each['topic']] += 1
            if topics[topic] > 1:
                topic =f'{topic}_{topics[topic]}'
                each['topic'] = topic
            out.append(each)
    if args.type == 'decomposition':
        output_path = args.input.replace('.jsonl', '_unique.jsonl')
    else:
        output_path = args.input.replace('.jsonl', f'_cleaned_{len(out)}.jsonl')
    write_jsonl(out, output_path)

