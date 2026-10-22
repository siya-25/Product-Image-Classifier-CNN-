"""
Data download module for Flipkart product images.
Downloads dataset from Hugging Face, parses categories, and downloads images.
"""

import os
import json
import ast
import logging
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse
from tqdm import tqdm
import requests
from PIL import Image
import io
import hashlib
import random
import numpy as np
from datasets import load_dataset
import pandas as pd
import imagehash

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _parse_image_urls(value) -> List[str]:
    """Parse JSON or Python-style image URL lists, ignoring invalid entries."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            image_urls = json.loads(value)
        except json.JSONDecodeError:
            try:
                image_urls = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
    else:
        image_urls = value

    if not isinstance(image_urls, (list, tuple)):
        return []

    urls = []
    for candidate in image_urls:
        if not isinstance(candidate, str):
            continue
        url = candidate.strip()
        try:
            parsed = urlparse(url)
        except ValueError:
            continue
        if parsed.scheme in ("http", "https") and parsed.netloc:
            urls.append(url)
    return urls


class FlipkartDataDownloader:
    """Downloads and processes Flipkart product images from Hugging Face."""
    
    def __init__(
        self,
        output_dir: str = "data",
        label_map_path: str = "configs/label_map.yaml",
        target_per_class: int = 1500,
        max_ratio: float = 2.0,
        image_size: Tuple[int, int] = (224, 224),
        seed: int = 42
    ):
        """
        Initialize downloader.
        
        Args:
            output_dir: Directory to save processed data
            label_map_path: Path to label mapping YAML
            target_per_class: Target images per class
            max_ratio: Maximum ratio between classes
            image_size: Target image size for resizing
            seed: Random seed for reproducibility
        """
        self.output_dir = Path(output_dir)
        self.label_map_path = Path(label_map_path)
        self.target_per_class = target_per_class
        self.max_ratio = max_ratio
        self.image_size = image_size
        self.seed = seed
        
        # Set random seed for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        
        # Create directories
        self.raw_dir = self.output_dir / "raw"
        self.processed_dir = self.output_dir / "processed"
        self.cache_dir = self.output_dir / "cache"
        
        for dir_path in [self.raw_dir, self.processed_dir, self.cache_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Load label map
        with open(self.label_map_path, 'r') as f:
            self.label_map = yaml.safe_load(f)
        
        # Create reverse mapping (keywords to class)
        self.keyword_to_class = {}
        for class_name, keywords in self.label_map.items():
            for keyword in keywords:
                self.keyword_to_class[keyword.lower()] = class_name
        
        logger.info(f"Initialized downloader with {len(self.keyword_to_class)} category keywords")
    
    def load_huggingface_dataset(self) -> pd.DataFrame:
        """Load Flipkart dataset from Hugging Face."""
        logger.info("Loading dataset from Hugging Face: jason1966/PromptCloudHQ_flipkart-products")
        
        try:
            dataset = load_dataset("jason1966/PromptCloudHQ_flipkart-products", split="train")
            df = dataset.to_pandas()
            logger.info(f"Loaded {len(df)} products from Hugging Face")
            return df
        except Exception as e:
            logger.error(f"Failed to load dataset from Hugging Face: {e}")
            raise
    
    def parse_category(self, category_tree: str) -> Optional[str]:
        """
        Parse product category tree and map to our 3 classes.
        
        Args:
            category_tree: Category tree string from dataset
            
        Returns:
            Mapped class name or None if no match
        """
        if pd.isna(category_tree) or not category_tree:
            return None
        
        # Parse the category tree (it's a JSON-like string)
        try:
            # Remove brackets and quotes, split by >>
            categories = [cat.strip().lower() for cat in category_tree.replace('["', '').replace('"]', '').split('>>')]
            
            # Find first matching keyword
            for category in categories:
                for keyword, class_name in self.keyword_to_class.items():
                    if keyword in category:
                        return class_name
        except Exception as e:
            logger.debug(f"Failed to parse category tree: {category_tree}, error: {e}")
        
        return None
    
    def download_image(
        self,
        url: str,
        save_path: Path,
        max_retries: int = 3,
        timeout: int = 10
    ) -> bool:
        """
        Download a single image with retries.
        
        Args:
            url: Image URL
            save_path: Path to save image
            max_retries: Number of retry attempts
            timeout: Request timeout in seconds
            
        Returns:
            True if successful, False otherwise
        """
        for attempt in range(max_retries):
            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()
                
                # Verify it's an image
                img = Image.open(io.BytesIO(response.content))
                img.verify()
                
                # Reopen for saving (verify closes the file)
                img = Image.open(io.BytesIO(response.content))
                
                # Convert to RGB if necessary
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                
                # Resize
                img = img.resize(self.image_size, Image.LANCZOS)
                
                # Save
                img.save(save_path, 'JPEG', quality=85)
                return True
                
            except Exception as e:
                logger.debug(f"Attempt {attempt + 1} failed for {url}: {e}")
                if attempt == max_retries - 1:
                    return False
        
        return False
    
    def compute_phash(self, image_path: Path) -> Optional[str]:
        """Compute perceptual hash for deduplication."""
        try:
            img = Image.open(image_path)
            return str(imagehash.phash(img))
        except Exception as e:
            logger.debug(f"Failed to compute phash for {image_path}: {e}")
            return None
    
    def deduplicate_images(
        self,
        image_paths: List[Path],
        class_name: str
    ) -> List[Path]:
        """
        Remove duplicate images using perceptual hashing.
        
        Args:
            image_paths: List of image paths
            class_name: Class name for logging
            
        Returns:
            List of unique image paths
        """
        logger.info(f"Deduplicating {len(image_paths)} images for class {class_name}")
        
        phash_to_path = {}
        unique_paths = []
        
        for img_path in tqdm(image_paths, desc=f"Deduplicating {class_name}"):
            phash = self.compute_phash(img_path)
            if phash and phash not in phash_to_path:
                phash_to_path[phash] = img_path
                unique_paths.append(img_path)
            elif phash:
                # Duplicate found, delete it
                img_path.unlink()
        
        logger.info(f"Removed {len(image_paths) - len(unique_paths)} duplicates for {class_name}")
        return unique_paths
    
    def process_dataset(self, df: pd.DataFrame) -> Dict[str, List[Path]]:
        """
        Process the dataset: map categories, download images, deduplicate.
        
        Args:
            df: Raw DataFrame from Hugging Face
            
        Returns:
            Dictionary mapping class names to lists of image paths
        """
        # Map categories
        logger.info("Mapping product categories to 3 classes")
        df['mapped_class'] = df['product_category_tree'].apply(self.parse_category)
        
        # Filter out unmapped products
        df = df[df['mapped_class'].notna()].copy()
        logger.info(f"Mapped {len(df)} products to classes")
        
        # Count per class
        class_counts = df['mapped_class'].value_counts()
        logger.info(f"Class distribution before download:\n{class_counts}")
        
        # Download images per class
        class_images = {}
        
        for class_name in self.label_map.keys():
            class_df = df[df['mapped_class'] == class_name]
            
            if len(class_df) == 0:
                logger.warning(f"No products found for class {class_name}")
                continue
            
            # Limit to target * max_ratio to avoid excessive downloads
            max_samples = min(len(class_df), self.target_per_class * int(self.max_ratio))
            class_df = class_df.head(max_samples)
            
            logger.info(f"Downloading up to {len(class_df)} images for class {class_name}")
            
            # Create class directory
            class_dir = self.processed_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            # Parse image URLs (they're stored as JSON strings)
            image_paths = []
            sample_idx = 0
            
            for row_idx, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Downloading {class_name}"):
                try:
                    # Parse image URLs
                    image_urls = _parse_image_urls(row['image'])
                    if not image_urls:
                        continue
                    
                    # Use first URL
                    url = image_urls[0]
                    
                    # Generate filename using stable counter
                    filename = f"{class_name}_{sample_idx:05d}.jpg"
                    save_path = class_dir / filename
                    sample_idx += 1
                    
                    # Skip if already exists
                    if save_path.exists():
                        image_paths.append(save_path)
                        continue
                    
                    # Download
                    if self.download_image(url, save_path):
                        image_paths.append(save_path)
                    else:
                        logger.debug(f"Failed to download: {url}")
                        
                except Exception as e:
                    logger.debug(f"Failed to process row {row_idx}: {e}")
            
            logger.info(f"Downloaded {len(image_paths)} images for {class_name}")
            
            # Deduplicate
            if image_paths:
                image_paths = self.deduplicate_images(image_paths, class_name)
            
            class_images[class_name] = image_paths
        
        return class_images
    
    def balance_classes(
        self,
        class_images: Dict[str, List[Path]]
    ) -> Dict[str, List[Path]]:
        """
        Balance classes by sampling to match target counts.
        
        Args:
            class_images: Dictionary of class to image paths
            
        Returns:
            Balanced dictionary
        """
        logger.info("Balancing classes")
        
        # Find minimum count
        counts = {cls: len(paths) for cls, paths in class_images.items()}
        min_count = min(counts.values())
        
        logger.info(f"Class counts before balancing: {counts}")
        
        # Use minimum count as target (or target_per_class if higher)
        target = min(min_count, self.target_per_class)
        
        balanced = {}
        for class_name, paths in class_images.items():
            if len(paths) > target:
                # Random sample with seed for reproducibility
                balanced[class_name] = random.sample(paths, target)
            else:
                balanced[class_name] = paths
        
        final_counts = {cls: len(paths) for cls, paths in balanced.items()}
        logger.info(f"Class counts after balancing: {final_counts}")
        
        return balanced
    
    def save_metadata(self, class_images: Dict[str, List[Path]]):
        """Save metadata and the exact balanced sample manifest."""
        # Create consistent class_to_idx mapping (sorted for reproducibility)
        class_names = sorted(class_images.keys())
        class_to_idx = {cls: idx for idx, cls in enumerate(class_names)}
        class_counts = {cls: len(class_images[cls]) for cls in class_names}
        if (
            not class_counts
            or 0 in class_counts.values()
            or len(set(class_counts.values())) != 1
        ):
            raise ValueError(f"Cannot save an unbalanced dataset manifest: {class_counts}")

        manifest_samples = [
            {
                'path': Path(image_path).resolve().relative_to(self.processed_dir.resolve()).as_posix(),
                'label': class_name
            }
            for class_name in class_names
            for image_path in class_images[class_name]
        ]
        for sample in manifest_samples:
            if not (self.processed_dir / sample['path']).is_file():
                raise FileNotFoundError(
                    f"Cannot save manifest; selected image is missing: "
                    f"{self.processed_dir / sample['path']}"
                )

        manifest_fingerprint = hashlib.sha256(
            json.dumps(manifest_samples, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        manifest = {
            'class_to_idx': class_to_idx,
            'class_counts': class_counts,
            'samples': manifest_samples,
            'fingerprint': manifest_fingerprint
        }
        manifest_path = self.processed_dir / 'manifest.json'
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        metadata = {
            'total_images': sum(class_counts.values()),
            'class_counts': class_counts,
            'classes': class_names,
            'class_to_idx': class_to_idx,
            'image_size': self.image_size,
            'target_per_class': self.target_per_class,
            'seed': self.seed
        }
        
        metadata_path = self.processed_dir / 'metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved metadata to {metadata_path}")
    
    def run(self):
        """Run the complete download pipeline."""
        logger.info("Starting data download pipeline")
        
        # Load dataset
        df = self.load_huggingface_dataset()
        
        # Process and download
        class_images = self.process_dataset(df)
        
        # Balance classes
        class_images = self.balance_classes(class_images)
        
        # Save metadata
        self.save_metadata(class_images)
        
        logger.info("Data download pipeline completed")
        
        return class_images


def main():
    """Main entry point."""
    downloader = FlipkartDataDownloader(
        output_dir="data",
        label_map_path="configs/label_map.yaml",
        target_per_class=1500,
        max_ratio=2.0,
        image_size=(224, 224),
        seed=42
    )
    
    class_images = downloader.run()
    
    # Print summary
    print("\n" + "="*50)
    print("DATASET DOWNLOAD SUMMARY")
    print("="*50)
    for class_name, paths in class_images.items():
        print(f"{class_name}: {len(paths)} images")
    print(f"Total: {sum(len(paths) for paths in class_images.values())} images")
    print("="*50)


if __name__ == "__main__":
    main()
