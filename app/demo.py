"""
Gradio demo app for product image classification.
"""

import gradio as gr
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import yaml
from pathlib import Path
import logging

from src.model.cnn import get_model
from torchvision import transforms

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductClassifierDemo:
    """Gradio demo for product image classification."""
    
    def __init__(
        self,
        checkpoint_path: str = "outputs/checkpoints/best_checkpoint.pth",
        config_path: str = "configs/config.yaml",
        confidence_threshold: float = 0.8
    ):
        """
        Initialize demo.
        
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to configuration file
            confidence_threshold: Threshold for auto-accept
        """
        # Load config
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.class_to_idx = checkpoint['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        # Initialize model
        self.model = get_model(
            model_type=self.config['model']['type'],
            num_classes=self.config['model']['num_classes'],
            dropout_rate=self.config['model']['dropout_rate'],
            pretrained=False,
            freeze_backbone=False
        ).to(self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        self.confidence_threshold = confidence_threshold
        self.image_size = self.config['data']['image_size']
        
        # Setup transforms
        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info(f"Loaded model from {checkpoint_path}")
        logger.info(f"Classes: {self.idx_to_class}")
        logger.info(f"Confidence threshold: {self.confidence_threshold}")
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> tuple:
        """
        Predict class for an image.
        
        Args:
            image: PIL Image
            
        Returns:
            Tuple of (prediction, confidence, flag)
        """
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Predict
        output = self.model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        
        # Get top prediction
        confidence, pred_idx = torch.max(probabilities, 0)
        confidence = confidence.item()
        pred_class = self.idx_to_class[pred_idx.item()]
        
        # Flag for manual review
        needs_review = confidence < self.confidence_threshold
        
        # Format output
        flag_text = "⚠️ MANUAL REVIEW" if needs_review else "✅ AUTO-ACCEPT"
        
        # Create confidence dictionary
        conf_dict = {
            self.idx_to_class[i]: probabilities[i].item()
            for i in range(len(probabilities))
        }
        
        return pred_class, confidence, flag_text, conf_dict
    
    def launch(self, share: bool = False):
        """Launch Gradio interface."""
        with gr.Blocks(title="Product Image Classifier") as interface:
            gr.Markdown("# Product Image Classifier")
            gr.Markdown("Upload a product image to classify it into Apparel, Electronics, or Home.")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(type="pil", label="Upload Image")
                    predict_btn = gr.Button("Classify", variant="primary")
                
                with gr.Column():
                    output_class = gr.Textbox(label="Predicted Class")
                    output_confidence = gr.Number(label="Confidence", precision=4)
                    output_flag = gr.Textbox(label="Status")
                    output_probs = gr.Label(label="Class Probabilities")
            
            # Examples
            gr.Examples(
                examples=[],
                inputs=input_image,
                label="Example Images (add your own)"
            )
            
            # Bind click event
            predict_btn.click(
                fn=self.predict,
                inputs=input_image,
                outputs=[output_class, output_confidence, output_flag, output_probs]
            )
        
        interface.launch(share=share)


def main():
    """Main entry point."""
    demo = ProductClassifierDemo(
        checkpoint_path="outputs/checkpoints/best_checkpoint.pth",
        config_path="configs/config.yaml",
        confidence_threshold=0.8
    )
    
    demo.launch(share=False)


if __name__ == "__main__":
    main()
