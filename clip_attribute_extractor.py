import torch
import clip
from PIL import Image
from transformers import BertTokenizer, BertModel

class CLIPAttributeExtractor:
    """
    Extracts visual attributes (upper/lower color, accessories) from an image using CLIP,
    then creates a text description and encodes it using Sentence-BERT/BERT.
    """
    def __init__(self, device="cuda", bert_model_path="/home/tq_naeem/Project/.venv/main/Sentence/"):
        self.device = device
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        # Attribute templates and vocab
        self.attribute_templates = {
            'upper': ['{color} upper clothing', '{color} shirt', '{color} jacket'],
            'lower': ['{color} pants', '{color} trousers', '{color} shorts'],
            'accessory': ['carrying {item}', 'wearing {item}', 'with {item}']
        }
        self.colors = ['red', 'blue', 'black', 'white', 'gray', 'green', 'yellow', 'pink']
        self.accessories = ['backpack', 'bag', 'hat', 'umbrella', 'none']

        # For attribute text embedding (Sentence-BERT/BERT)
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_path, local_files_only=True)
        self.bert_model = BertModel.from_pretrained(bert_model_path, local_files_only=True).to(device)
    
    def _generate_prompts(self):
        prompts = []
        for color in self.colors:
            for template in self.attribute_templates['upper']:
                prompts.append(template.format(color=color))
        for color in self.colors:
            for template in self.attribute_templates['lower']:
                prompts.append(template.format(color=color))
        for item in self.accessories:
            if item == 'none': continue
            for template in self.attribute_templates['accessory']:
                prompts.append(template.format(item=item))
        return prompts

    def extract_attributes(self, image_path):
        """
        Returns a dictionary: {'upper': color, 'lower': color, 'accessory': item}
        """
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.preprocess(image).unsqueeze(0).to(self.device)
            text = clip.tokenize(self._generate_prompts()).to(self.device)

            with torch.no_grad():
                image_features = self.model.encode_image(image)
                text_features = self.model.encode_text(text)
                similarity = (image_features @ text_features.T).softmax(dim=-1)
                probs = similarity.cpu().numpy()[0]

            attributes = {'upper': 'unknown', 'lower': 'unknown', 'accessory': 'none'}
            upper_probs = probs[:24]
            if upper_probs.sum() > 0:
                upper_color_idx = upper_probs.argmax() // 3
                attributes['upper'] = self.colors[upper_color_idx]
            lower_probs = probs[24:48]
            if lower_probs.sum() > 0:
                lower_color_idx = lower_probs.argmax() // 3
                attributes['lower'] = self.colors[lower_color_idx]
            acc_probs = probs[48:]
            if acc_probs.max() > 0.3 and acc_probs.sum() > 0:
                acc_idx = acc_probs.argmax()
                attributes['accessory'] = self.accessories[acc_idx // 3]
            return attributes
        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")
            return {'upper': 'unknown', 'lower': 'unknown', 'accessory': 'none'}

    def attributes_to_text(self, attributes):
        """Convert dict to a natural language description."""
        desc = f"A person wearing {attributes['upper']} top and {attributes['lower']} bottoms"
        if attributes['accessory'] != 'none':
            desc += f" with {attributes['accessory']}"
        return desc

    def encode_attributes(self, attributes):
        """
        Convert attributes dict to text, then to a 768-dim embedding using BERT.
        """
        text = self.attributes_to_text(attributes)
        inputs = self.bert_tokenizer(text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            emb = outputs.last_hidden_state[:, 0, :]  # CLS token, shape [1, 768]
        return emb.squeeze(0).cpu().numpy()  # [768,]