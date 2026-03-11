"""分析二十四史语料的字符统计"""
import os, glob
from collections import Counter

chars = set()
total = 0
counter = Counter()
for f in glob.glob(r'c:\Users\97290\Desktop\MOE\二十四史txt\*.txt'):
    with open(f, 'r', encoding='gb18030', errors='ignore') as fh:
        text = fh.read()
        chars.update(text)
        counter.update(text)
        total += len(text)

print(f'总字符数: {total:,}')
print(f'唯一字符数: {len(chars):,}')
print(f'\n最常见20个字符: {counter.most_common(20)}')
print(f'出现次数>=10的字符数: {sum(1 for c,n in counter.items() if n>=10)}')
print(f'出现次数>=100的字符数: {sum(1 for c,n in counter.items() if n>=100)}')
print(f'出现次数>=1000的字符数: {sum(1 for c,n in counter.items() if n>=1000)}')

# 估算GPT-2 BPE token数
try:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    sample_file = glob.glob(r'c:\Users\97290\Desktop\MOE\二十四史txt\*.txt')[0]
    with open(sample_file, 'r', encoding='gb18030', errors='ignore') as fh:
        sample = fh.read()[:10000]
    tokens = enc.encode(sample)
    ratio = len(tokens) / len(sample)
    print(f'\nGPT-2 BPE token/char比率(采样): {ratio:.2f}')
    print(f'GPT-2 BPE估算总token数: {int(total * ratio):,}')
except:
    print('\ntiktoken未安装，跳过BPE分析')
