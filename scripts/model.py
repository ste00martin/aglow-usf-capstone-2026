"""
FINAL FIXED model.py for HuggingFace
This version works with standard 'image-classification' pipeline
No custom pipeline registration needed
"""

import torch
import torch.nn as nn
from transformers import ViTModel, ViTPreTrainedModel
from transformers.modeling_outputs import ImageClassifierOutput


class AgeGenderViTModel(ViTPreTrainedModel):
    """
    Age-Gender Vision Transformer Model
    Architecture: ViT-Base with dual heads for age and gender prediction
    """
    
    def __init__(self, config):
        super().__init__(config)
        self.vit = ViTModel(config, add_pooling_layer=False)
        
        # Age regression head: 768 -> 256 -> 64 -> 1
        self.age_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(64, 1)
        )
        
        # Gender classification head: 768 -> 256 -> 64 -> 1 (sigmoid)
        self.gender_head = nn.Sequential(
            nn.Linear(config.hidden_size, 256), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(256, 64), 
            nn.ReLU(), 
            nn.Dropout(0.1),
            nn.Linear(64, 1), 
            nn.Sigmoid()
        )
        
        # Classifier for pipeline compatibility
        self.num_labels = 2
        self.config.num_labels = 2
        self.classifier = nn.Linear(config.hidden_size, 2)
        self.post_init()
        
    def forward(self, pixel_values=None, labels=None, **kwargs):
        outputs = self.vit(pixel_values=pixel_values, **kwargs)
        sequence_output = outputs[0]
        pooled_output = sequence_output[:, 0]  # CLS token
        
        age_output = self.age_head(pooled_output)
        gender_output = self.gender_head(pooled_output)
        
        # Concatenate age and gender for custom processing
        logits = torch.cat([age_output, gender_output], dim=1)
        
        # Store in output for postprocessing
        return ImageClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states if hasattr(outputs, 'hidden_states') else None,
            attentions=outputs.attentions if hasattr(outputs, 'attentions') else None,
        )


# Helper function for direct usage (bypassing pipeline)
def predict_age_gender(image_path):
    from transformers import AutoConfig, AutoImageProcessor
    from PIL import Image
    import requests
    from io import BytesIO
    import torch
    from model import AgeGenderViTModel  # if defined in model.py
    # If not, see note below

    model_name = "abhilash88/age-gender-prediction"

    config = AutoConfig.from_pretrained(
        model_name,
        trust_remote_code=True
    )

    model = AgeGenderViTModel.from_pretrained(
        model_name,
        config=config,
        trust_remote_code=True
    )

    model.eval()

    processor = AutoImageProcessor.from_pretrained(model_name)

    # Load image
    if isinstance(image_path, str):
        if image_path.startswith(('http://', 'https://')):
            response = requests.get(image_path)
            image = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

    # 🔥 FIX: explicitly define crop_size
    inputs = processor(
        images=image,
        return_tensors="pt",
        do_center_crop=True,
        crop_size={"height": 224, "width": 224}
    )

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits[0]

    age = int(round(logits[0].item()))
    age = max(0, min(100, age))

    gender_prob_female = torch.sigmoid(logits[1]).item()
    gender_prob_male = 1.0 - gender_prob_female

    gender = "Female" if gender_prob_female >= 0.5 else "Male"
    confidence = max(gender_prob_female, gender_prob_male)

    return {
        'age': age,
        'gender': gender,
        'gender_confidence': float(confidence),
        'gender_probability_male': float(gender_prob_male),
        'gender_probability_female': float(gender_prob_female),
        'label': f"{age} years, {gender}",
        'score': float(confidence)
    }


def simple_predict(image_path):
    """
    Simple prediction with string output
    
    Args:
        image_path: Path to image
    
    Returns:
        String: "25 years, Female (87% confidence)"
    """
    result = predict_age_gender(image_path)
    return f"{result['age']} years, {result['gender']} ({result['gender_confidence']:.1%} confidence)"

