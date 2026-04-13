import torch
import json
from transformers import BertTokenizer, BertForSequenceClassification
from openai import OpenAI
import os

# ── Load model once at module level ──────────────────────────────────────
MODEL_PATH          = './models/bert_requirements_final'
AMBIGUITY_THRESHOLD = 0.55
SECOND_OPINION_THRESHOLD = 0.75

device    = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model     = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model     = model.to(device)
model.eval()

with open(f'{MODEL_PATH}/label2id.json') as f:
    label2id = json.load(f)
with open(f'{MODEL_PATH}/id2label.json') as f:
    id2label = {int(k): v for k, v in json.load(f).items()}

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AGENT_MODEL   = "gpt-4o"
LABELING_MODEL = "gpt-4o-mini"


def classify_requirement(text, threshold=AMBIGUITY_THRESHOLD):
    inputs = tokenizer(
        text, return_tensors='pt',
        truncation=True, padding='max_length', max_length=128
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs      = torch.softmax(outputs.logits, dim=1).squeeze()
    confidence = probs.max().item()
    pred_id    = probs.argmax().item()
    pred_label = id2label[pred_id]
    all_probs  = {id2label[i]: round(probs[i].item(), 4) for i in range(len(id2label))}

    if confidence < threshold:
        return 'Ambiguous', confidence, all_probs

    if confidence < SECOND_OPINION_THRESHOLD:
        gpt_label = gpt_second_opinion(text)
        if gpt_label:
            return gpt_label, confidence, all_probs

    return pred_label, confidence, all_probs


def gpt_second_opinion(text):
    valid_labels = list(label2id.keys())
    prompt = f"""Classify this software requirement into exactly one label:
{', '.join(valid_labels)}

Label definitions:
- FR: what the system SHALL DO
- NFR_Performance: speed, response time, throughput
- NFR_Security: authentication, encryption, access control
- NFR_Usability: ease of use, accessibility
- NFR_Reliability: availability, uptime, fault tolerance
- NFR_Maintainability: modularity, testability
- NFR_Scalability: capacity, load handling
- NFR_Portability: platform compatibility
- NFR_Operational: deployment, infrastructure
- NFR_Legal: compliance, regulations
- NFR_LookAndFeel: visual design, branding
- NFR_Other: fits no other category — use sparingly
- Ambiguous: vague, untestable

Requirement: "{text}"
Reply with ONLY the label."""

    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "Reply with one label only."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=20, temperature=0
        )
        label = response.choices[0].message.content.strip().replace('`', '')
        return label if label in valid_labels else None
    except:
        return None