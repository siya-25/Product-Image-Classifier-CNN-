import hashlib
import json
import logging
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from src.data.dataset import ProductImageDataset, get_data_loaders
from src.data.download import FlipkartDataDownloader, _parse_image_urls


def _create_dataset(data_dir: Path, images_per_class: int = 10) -> None:
    processed_dir = data_dir / "processed"
    classes = {"Apparel": 0, "Electronics": 1, "Home": 2}
    processed_dir.mkdir(parents=True)
    manifest_samples = []
    for class_name in classes:
        class_dir = processed_dir / class_name
        class_dir.mkdir()
        for index in range(images_per_class):
            Image.new("RGB", (8, 8), color=(index, 0, 0)).save(
                class_dir / f"{index}.jpg"
            )
            manifest_samples.append(
                {"path": f"{class_name}/{index}.jpg", "label": class_name}
            )
    counts = {class_name: images_per_class for class_name in classes}
    fingerprint = hashlib.sha256(
        json.dumps(manifest_samples, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    (processed_dir / "manifest.json").write_text(
        json.dumps({
            "class_to_idx": classes,
            "class_counts": counts,
            "samples": manifest_samples,
            "fingerprint": fingerprint,
        }),
        encoding="utf-8",
    )
    (processed_dir / "metadata.json").write_text(
        json.dumps({
            "class_to_idx": classes,
            "class_counts": counts,
            "total_images": len(manifest_samples),
        }),
        encoding="utf-8",
    )


def test_data_loaders_use_independent_datasets_and_reuse_valid_split(tmp_path):
    data_dir = tmp_path / "data"
    _create_dataset(data_dir)

    loaders = get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    train, val, test = loaders[:3]
    train_data, val_data, test_data = (
        train.dataset.dataset,
        val.dataset.dataset,
        test.dataset.dataset,
    )

    assert train_data is not val_data
    assert train_data is not test_data
    assert val_data is not test_data
    assert train_data.transform is not val_data.transform
    assert train_data.transform is not test_data.transform
    assert val_data.transform is not test_data.transform
    assert train_data.samples == val_data.samples == test_data.samples

    split_file = data_dir / "processed" / "split_indices.json"
    original_split = json.loads(split_file.read_text(encoding="utf-8"))
    get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    reused_split = json.loads(split_file.read_text(encoding="utf-8"))
    assert reused_split["train_idx"] == original_split["train_idx"]
    assert reused_split["val_idx"] == original_split["val_idx"]
    assert reused_split["test_idx"] == original_split["test_idx"]


def test_changed_dataset_invalidates_saved_split(tmp_path):
    data_dir = tmp_path / "data"
    _create_dataset(data_dir)
    get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    split_file = data_dir / "processed" / "split_indices.json"
    previous = json.loads(split_file.read_text(encoding="utf-8"))

    manifest_file = data_dir / "processed" / "manifest.json"
    manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    Image.new("RGB", (8, 8)).save(data_dir / "processed" / "Apparel" / "replacement.jpg")
    manifest["samples"][0]["path"] = "Apparel/replacement.jpg"
    manifest["fingerprint"] = hashlib.sha256(
        json.dumps(manifest["samples"], sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    refreshed = json.loads(split_file.read_text(encoding="utf-8"))

    assert refreshed["sample_fingerprint"] != previous["sample_fingerprint"]
    assert refreshed["total_samples"] == previous["total_samples"]
    assert sorted(
        refreshed["train_idx"] + refreshed["val_idx"] + refreshed["test_idx"]
    ) == list(range(refreshed["total_samples"]))

    refreshed["total_samples"] -= 1
    split_file.write_text(json.dumps(refreshed), encoding="utf-8")
    get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    repaired = json.loads(split_file.read_text(encoding="utf-8"))
    assert repaired["total_samples"] == len(repaired["train_idx"]) + len(
        repaired["val_idx"]
    ) + len(repaired["test_idx"])


def test_dataset_loads_only_manifest_samples_and_checks_integrity(tmp_path):
    data_dir = tmp_path / "data"
    _create_dataset(data_dir)
    excluded = data_dir / "processed" / "Apparel" / "unselected.jpg"
    Image.new("RGB", (8, 8)).save(excluded)

    train, _, _, _ = get_data_loaders(str(data_dir), num_workers=0, image_size=8)
    assert len(train.dataset.dataset) == 30
    assert all("unselected.jpg" not in path for path, _ in train.dataset.dataset.samples)

    manifest = json.loads(
        (data_dir / "processed" / "manifest.json").read_text(encoding="utf-8")
    )
    (data_dir / "processed" / manifest["samples"][0]["path"]).unlink()
    with pytest.raises(FileNotFoundError, match="Manifest image is missing"):
        ProductImageDataset(str(data_dir))


def test_dataset_rejects_inconsistent_manifest_class_counts(tmp_path):
    data_dir = tmp_path / "data"
    _create_dataset(data_dir)
    manifest_path = data_dir / "processed" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["class_counts"]["Apparel"] -= 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="Manifest class counts are inconsistent"):
        ProductImageDataset(str(data_dir))


def test_save_metadata_writes_balanced_manifest(tmp_path):
    downloader = object.__new__(FlipkartDataDownloader)
    downloader.processed_dir = tmp_path / "processed"
    downloader.processed_dir.mkdir()
    downloader.image_size = (8, 8)
    downloader.target_per_class = 1
    downloader.seed = 42
    class_images = {}
    for class_name in ("Apparel", "Electronics", "Home"):
        image_path = downloader.processed_dir / class_name / "selected.jpg"
        image_path.parent.mkdir()
        Image.new("RGB", (8, 8)).save(image_path)
        class_images[class_name] = [image_path]

    downloader.save_metadata(class_images)

    manifest = json.loads(
        (downloader.processed_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert len(manifest["samples"]) == 3
    assert manifest["class_counts"] == {
        "Apparel": 1,
        "Electronics": 1,
        "Home": 1,
    }


def test_image_url_parser_accepts_json_and_python_lists():
    assert _parse_image_urls('["https://example.com/a.jpg"]') == [
        "https://example.com/a.jpg"
    ]
    assert _parse_image_urls("['https://example.com/b.jpg']") == [
        "https://example.com/b.jpg"
    ]
    assert _parse_image_urls(
        "['', 'not a URL', 'http://[invalid', 'https://example.com/c.jpg']"
    ) == ["https://example.com/c.jpg"]
    assert _parse_image_urls("not a list") == []
    assert _parse_image_urls("[]") == []


def test_download_error_logs_actual_row_index(tmp_path, caplog):
    downloader = object.__new__(FlipkartDataDownloader)
    downloader.label_map = {"Apparel": ["Clothing"]}
    downloader.keyword_to_class = {"clothing": "Apparel"}
    downloader.processed_dir = tmp_path
    downloader.target_per_class = 10
    downloader.max_ratio = 2.0

    def fail_download(url, save_path):
        raise RuntimeError("simulated download failure")

    downloader.download_image = fail_download
    frame = pd.DataFrame(
        [{"product_category_tree": "Clothing", "image": "['https://example.com/a.jpg']"}],
        index=[17],
    )

    with caplog.at_level(logging.DEBUG):
        result = downloader.process_dataset(frame)

    assert result == {"Apparel": []}
    assert "Failed to process row 17" in caplog.text
