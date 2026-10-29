"""Reload the saved fine-tuned classifier for custom text inference."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

p=argparse.ArgumentParser()
root=Path(__file__).resolve().parent
corrected_models=sorted((root/'corrected').glob('*/checkpoints/hw4_corrected_final'))
default_model=str(corrected_models[0]) if corrected_models else str(root/'saved_bert')
p.add_argument('text',nargs='*',default=['I loved the GPU lab today!', 'This was a frustrating day.'])
p.add_argument('--model',default=default_model)
args=p.parse_args()
tokenizer=AutoTokenizer.from_pretrained(args.model)
model=AutoModelForSequenceClassification.from_pretrained(args.model).eval()
inputs=tokenizer(args.text or ['I loved the GPU lab today!'],return_tensors='pt',padding=True,truncation=True,max_length=128)
with torch.inference_mode(): probabilities=model(**inputs).logits.softmax(-1)
for text,probs in zip(args.text,probabilities):
    print({'text':text,'negative_probability':float(probs[0]),'positive_probability':float(probs[1])})
if Path(args.model).resolve()==(root/'saved_bert').resolve():
    print('Caution: this legacy baseline learned from an all-negative slice; predictions do not establish sentiment generalization.')
