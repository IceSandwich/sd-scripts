import argparse
import os
import typing
import json
from PIL import Image
import logging
log = logging.Logger("log", logging.INFO)

import library.mep as mep

def SetupParser(parser: argparse.ArgumentParser):
	parser.add_argument("--dataset", type=str, required=True)
	parser.add_argument("--output", type=str, required=True)
	parser.add_argument("--mep_key", type = str, required=True)

def main(args):
	mep.Init(args.mep_key)

	print("- Create directory: ", args.output)
	os.makedirs(args.output, exist_ok=True)

	# json: {`img_path`:{"caption": "caption...", "resolution": [width, height]}, ...}
	meta: typing.Dict[str, typing.Any] = {}
	
	print("- Loading dataset...")
	# 支持的图片扩展名
	image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
	folder_path: str = args.dataset
	# 遍历文件夹
	index = 1
	for filename in os.listdir(folder_path):
		name, ext = os.path.splitext(filename)

		# 检查是否为图片
		if ext.lower() not in image_extensions:
			continue

		image_path = os.path.join(folder_path, filename)
		txt_filename = f"{name}.txt"
		txt_path = os.path.join(folder_path, txt_filename)
		
		# 检查并读取对应的文本文件
		if os.path.exists(txt_path):
			with open(txt_path, 'r', encoding='utf-8') as f:
				prompt = f.read().strip()
		else:
			print("没有找到对应的txt文件：", name)
			raise Exception("Not found correspondent txt file.")
		
		img = Image.open(image_path)
		width = img.width
		height = img.height
		img.close()

		mep_img_path = f"dat{index}.mep"
		index = index + 1
		meta[mep_img_path] = {
			"caption": prompt,
			"resolution": [width, height],
			"fn": filename
		}

		mep.SaveImage(image_path, os.path.join(args.output, mep_img_path))
		print(f"-- Convert {image_path} => {mep_img_path}")

	print(f"- Converted {len(meta)} images.")
	print(f"- Sample of first image:", meta[list(meta.keys())[0]])

	mdFilename = os.path.join(args.output, mep.MetadataFilename)
	mep.SaveMeta(mdFilename, meta)
	print(f"- Metadata saved to {mdFilename}")

def validation(args):
	mep.Init(args.mep_key)

	mdFilename = os.path.join(args.output, mep.MetadataFilename)
	jsc = mep.ReadJSON(mdFilename)
	print(f"- Read images: ", len(jsc))

	key1 = list(jsc.keys())[0]
	print(f"- Sample of first image:", jsc[key1])

	img = mep.ReadImage(os.path.join(args.output, key1))
	imgFilename = os.path.join(args.output, jsc[key1]["fn"])
	img.save(imgFilename)
	print(f"- Sample of first image saved to", imgFilename)
	print(f"！！ You can safetly delete the sample of first image generated above before you update this dataset.")

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	SetupParser(parser)
	args = parser.parse_args()
	main(args)
	print()
	validation(args)
	print()
	print("Done.")