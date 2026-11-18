import gdown
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from sklearn.model_selection import train_test_split

CAPTIONS_FILE = "captions.jsonl"
DATA_OUTPUT_DIR = "data/artcap/"
SEED = 10623


def download_file(file_id, path):
    if gdown.download(id=file_id, output=str(path)) is None:
        raise RuntimeError(f"Could not download {path.name}")

def make_caption_rows(images_dir, train_val, test):
    image_paths = {}
    for path in images_dir.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"} or not path.is_file():
            continue
        elif "__MACOSX" in path.parts or path.name.startswith("._"):
            continue
        elif path.name in image_paths:
            raise ValueError(f"Duplicate image filename: {path.name}")
        else:
            image_paths[path.name] = path.relative_to(images_dir.parent).as_posix()

    # split the train_val set into train and val, keeping the same test split as Kale
    train, val = train_test_split(train_val, test_size=0.1, random_state=SEED)
    rows = []
    seen = set()
    for split, records in (("val", val), ("train", train), ("test", test)):
        for filename, captions in records:
            filename = Path(filename).name
            image_id = Path(filename).stem
            if image_id in seen or len(captions) != 5:
                raise ValueError(f"Duplicate image ID or wrong caption count: {image_id}")
            seen.add(image_id)
            rows.append({
                "image_id": image_id,
                "image_path": image_paths[filename],
                "split": split,
                "references": captions,
                "caption_ids": [f"{image_id}:{i}" for i in range(5)],
            })
    return rows

# helper function other files can call to load the ArtCap dataset - list of dictionaries
def load_artcap(split=None):
    if split not in (None, "train", "val", "test"):
        raise ValueError(f"Unknown split: {split}")

    rows = []
    data_dir = Path(DATA_OUTPUT_DIR)
    with (data_dir / CAPTIONS_FILE).open(encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if split is None or row["split"] == split:
                row["image_path"] = data_dir / row["image_path"]
                rows.append(row)
    return rows

def prepare_artcap():
    data_dir = Path(DATA_OUTPUT_DIR)
    captions_path = data_dir / CAPTIONS_FILE
    images_dir = data_dir / "images"

    # check if we already have the data, and if so, return early
    if captions_path.is_file() and images_dir.is_dir():
        print("Already downloaded and processed at: ", captions_path)
        return
    data_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=data_dir) as temp_dir:
        temp_dir = Path(temp_dir)

        # download ArtCap images if not present
        if not images_dir.is_dir():
            images_zip = temp_dir / "images.zip"
            download_file("1inQ4W-CtNhhfo1vAehSbLgyMPTmltWVa", images_zip)
            extracted = temp_dir / "images"
            with ZipFile(images_zip) as archive:
                archive.extractall(extracted)
            extracted.rename(images_dir)

        # download ArtCap captions if not present
        # we download Kale instead just so that we keep the same splits as the ones Kale used - captions are identical
        kale_zip = temp_dir / "data.zip"
        download_file("1OdPZMQQACgy2OmRsIp3-W-KnD7ATC-hO", kale_zip)
        with ZipFile(kale_zip) as archive:
            train_val = json.loads(archive.read("data/ArtCap/artcap_train_val.json"))
            test = json.loads(archive.read("data/ArtCap/artcap_test.json"))

        # create the captions file with the same splits as Kale
        rows = make_caption_rows(images_dir, train_val, test)
        captions_tmp = temp_dir / CAPTIONS_FILE
        with captions_tmp.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
        captions_tmp.replace(captions_path)

    print("Captions saved to: ", captions_path)


if __name__ == "__main__":
    prepare_artcap()