import argparse
import pathlib


parser = argparse.ArgumentParser()
parser.add_argument("--folders", nargs='+', help="path to dataset")
parser.add_argument("--output", default="/private/home/rdessi/imagenet_paths.txt")

opts = parser.parse_args()
paths = list(dict.fromkeys(opts.folders))  # remove duplicates from paths to folders

with open(opts.output, 'w') as fout:
    for path in paths:
        path = pathlib.Path(path).resolve()
        for filename in path.glob("**/*.JPEG"):
            fout.write(f"{path / filename}\n")
