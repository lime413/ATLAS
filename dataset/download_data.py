import os
import requests
from tqdm.auto import tqdm

urls = [
    "http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl",
    "http://dl.fbaipublicfiles.com/BLINK/enwiki-pages-articles.xml.bz2"
]

# target directory
download_dir = r"E:\\ir-research-natural-questions"
os.makedirs(download_dir, exist_ok=True)

for url in urls:
    base = url.split("/")[-1]
    filename = os.path.join(download_dir, base)

    r = requests.get(url, stream=True)
    total_size = int(r.headers.get("content-length", 0))
    block_size = 1024

    t = tqdm(total=total_size, unit="iB", unit_scale=True, desc=base)

    with open(filename, "wb") as f:
        for data in r.iter_content(block_size):
            t.update(len(data))
            f.write(data)

    t.close()