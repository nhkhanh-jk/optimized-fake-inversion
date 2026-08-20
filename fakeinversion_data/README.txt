# Public Version of FakeInversion Dataset

The code below and data provided in this zip file are distributed under CC BY 4.0 license.

Original work performed at Google Research used a different set of prompts and text-to-image models. To accelerate responsible research in this societally important field, this file provides instructions for recreating an alternative version of this benchmark using publicly available text-to-image models and datasets.

URLs of images and models used below might eventually become unavailable or point to different data. These URLs are provided as is - in no event shall the authors or copyright holders be liable for any claim, damages or other liability, whether in an action of contract, tort or otherwise, arising from, out of or in connection with this data.

First, clone this external repo and install dependencies:
```
git clone github.com/georgecazenavette/easy-diffusion-generation.git
conda create -n fake-inversion
conda install pip
pip install -r easy-diffusion-genetation/requirements.txt
```

To build a list of prompts, please run the following:

```
cat << EOF > build_prompts.py
import datasets

dataset = datasets.load_dataset("wanng/midjourney-v5-202304-clean", split="train").filter(lambda example: example["upscaled"]).shuffle(seed=42)

prompts = [row['clean_prompts'] for row in dataset.select(range(5_000))]

with open("fake_inversion_prompts.txt", "w") as f:
    f.write("\n".join(prompts))
EOF

python build_prompts.py
mv fake_inversion_prompts.txt ./easy-diffusion-generation
cd easy-diffusion-generation
```

To generate synthetic images using publicly available models, please run:
```
for m in kandinsky2 kandinsky3 pixart-alpha playground-25 sd-15 sd-21 sdxl-dpo sdxl ssd1b stable-cascade vega wurstchen2; do python gen.py --prompts=fake_inversion --det_seet --model=${m}; done
```

To fetch DALLE images, please run:
```
cat << EOF > fetch_dalle.py
import os
import tqdm
import datasets

dataset = datasets.load_dataset("OpenDatasets/dalle-3-dataset", ignore_verifications=True, revision="22a1f7dc2ea1137ec5608bf791e70937b6a4df78")

ds = dataset['train'].shard(num_shards=5, index=0)

os.makedirs("dalle3", exist_ok=True)
for i, row in tqdm.tqdm(enumerate(ds.select(range(3_000))), total=3_000):
    im = row['image']
    im.save(os.path.join("dalle3", "{:05d}.png".format(i)))
EOF

python fetch_dalle.py
```

To fetch MJ images, please run:
```
wget https://huggingface.co/datasets/ehristoforu/midjourney-images/resolve/main/Midjourney-dataset.zip?download=true
unzip Midjourney-dataset.zip\?download\=true

mkdir mj_all mj_big
mv ./photo*jpg ./mj_all

cat << EOF > fetch_mj.py
import glob
import shutil

for f in (set(glob.glob("mj_all/*.jpg")) - set(glob.glob("mj_all/*thumb*.jpg"))):
    shutil.move("{f}", "mj_big")
EOF

python fetch_mj.py
rm -r mj_all
mv mj_big mj
```

To fetch corresponding real images, unzip `fakeinversion_data.zip` and run:
```
cat << EOF > fetch_reals.py
import diffusers, json, os, glob, tqdm

json_files = list(sorted(glob.glob("url_files/*.jsonl")))

total = 0
failed = 0

for file in json_files:
    dataset = os.path.basename(file)[:-6]
    print("Fetching {}...".format(dataset))
    save_dir = "fetched_images/{}".format(dataset)
    os.makedirs(save_dir, exist_ok=True)
    with open(file, 'r') as json_file:
        json_list = list(json_file)

    json_list = list(sorted(json_list, key=lambda x: int(json.loads(x)["prompt_index"])))

    for json_str in tqdm.tqdm(json_list):
        result = json.loads(json_str)
        total += 1
        url = result["url"]
        prompt_index = int(result["prompt_index"])
        save_path = os.path.join(save_dir, "{:05d}.png".format(prompt_index))
        if os.path.exists(save_path):
            print("Skipping existing image at {}".format(save_path))

        try:
            im = diffusers.utils.load_image(url)
            im.verify()
        except:
            failed += 1
            print("Falied to download image {} from {}".format(prompt_index, url))
            print("Failure rate: {}".format(failed / total))
            continue

        im.save(save_path)
EOF

python fetch_reals.py
```

NOTE: Facebook, Instagram and Imgur image URLs are ephemeral, so we provide URLs to public WEB PAGES containing these images. End users will need to fetch image bytes themselves via other means.