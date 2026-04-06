import json
import time
from openai import OpenAI
import os
from agent.classifier import classify_requirement, gpt_second_opinion

client       = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
AGENT_MODEL  = "gpt-4o"
MAX_ITERATIONS = 3


def analyze_ambiguity(text):
    prompt = f"""Analyze this requirement for ambiguity: "{text}"

Respond ONLY in JSON:
{{
  "missing_properties": ["property1"],
  "issues": {{
    "Actor": "clear or issue",
    "Action": "clear or issue",
    "Object": "clear or issue",
    "Measurable criterion": "clear or issue"
  }},
  "ambiguity_severity": "high/medium/low",
  "summary": "one sentence"
}}"""
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "Respond only in valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=400, temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"missing_properties": [], "issues": {}, 
                "ambiguity_severity": "medium", "summary": "Analysis failed"}


def generate_clarification_questions(text, analysis):
    severity      = analysis.get('ambiguity_severity', 'medium')
    max_questions = {'high': 3, 'medium': 2, 'low': 1}.get(severity, 2)
    prompt = f"""Generate {max_questions} clarification question(s) for:
"{text}"

Issues: {json.dumps(analysis.get('issues', {}), indent=2)}

Respond ONLY in JSON:
{{"questions": [{{"targets": "property", "question": "question text"}}]}}"""
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "Respond only in valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=300, temperature=0.2,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content).get('questions', [])
    except:
        return [{"targets": "Action", 
                 "question": "Could you describe more specifically what the system should do?"}]


def refine_requirement(original, questions, answers):
    qa = "\n".join([f"Q: {q['question']}\nA: {a}" 
                    for q, a in zip(questions, answers)])
    prompt = f"""Rewrite this vague requirement into a clear IEEE 830 requirement:
Original: "{original}"
Clarifications: {qa}

Respond ONLY in JSON:
{{"refined_requirement": "The system shall...", "improvements": ["improvement1"]}}"""
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "Respond only in valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=300, temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except:
        return {"refined_requirement": original, "improvements": []}


def auto_refine_ambiguous(text, label):
    prompt = f"""Rewrite this vague requirement as a clear, specific, testable requirement:
"{text}" (Category: {label})

Format: "The system shall [action] [object] [measurable condition]"
Remove all vague terms. Add measurable criteria.

Respond ONLY in JSON:
{{"refined_requirement": "...", "assumption": "..."}}"""
    try:
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=[
                {"role": "system", "content": "Respond only in valid JSON."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens=200, temperature=0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result.get('refined_requirement', text), result.get('assumption', '')
    except:
        return text, ''

def check_and_fix_requirement(text):
    """
    Check requirement for grammar errors and missing components.
    Fix grammar mistakes and flag missing actors.
    Returns (fixed_text, was_changed, issues_found)
    """
    prompt = f"""You are a requirements engineering expert and grammar checker.

Analyze this software requirement:
"{text}"

Check for:
1. Grammar mistakes (e.g. "they system" → "the system", "teh" → "the")
2. Missing actor — who performs the action? (e.g. "They should" → who is "they"?)
3. Typos or unclear pronouns

If there are issues:
- Fix grammar mistakes and typos automatically
- Replace vague pronouns ("they", "it", "users") with the most likely specific actor
  (e.g. "The user", "The administrator", "The system")
- Keep the original intent intact

Respond ONLY in JSON:
{{
  "fixed_text"   : "the corrected requirement",
  "was_changed"  : true or false,
  "issues_found" : ["issue 1", "issue 2"]
}}"""

    try:
        response = client.chat.completions.create(
            model       = AGENT_MODEL,
            messages    = [
                {"role": "system", "content": "You are a requirements quality checker. Respond only in JSON."},
                {"role": "user",   "content": prompt}
            ],
            max_tokens          = 200,
            temperature         = 0,
            response_format     = {"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return (
            result.get('fixed_text',    text),
            result.get('was_changed',   False),
            result.get('issues_found',  [])
        )
    except Exception as e:
        return text, False, []

def split_and_validate_input(text):
    """
    Before classification, check if the input contains:
    1. Multiple requirements in one sentence — split them
    2. Vague requirements — flag for clarification
    3. Grammar issues — fix them

    Returns a list of individual requirement strings to process separately.
    """
    prompt = f"""You are a requirements engineering expert.

Analyze this stakeholder input:
"{text}"

Perform these tasks:
1. SPLIT: If the input contains multiple requirements (joined by "and", ".", ";",
   or other conjunctions), split them into individual requirements.
   Each requirement should describe ONE system property or behavior only.

2. VALIDATE: For each split requirement, check if it is:
   - Clear and testable → keep as-is
   - Vague (missing measurable criteria, actor, or specific action) → flag as vague
   - Has grammar issues → fix them

3. Return each requirement as a separate item.

Examples of inputs needing splitting:
- "The system should be fast and easy to use." → two requirements
- "The system shall encrypt passwords. Users must be able to reset their password." → two requirements
- "It should work well and handle everything properly." → two requirements

Respond ONLY in JSON:
{{
  "requirements": [
    {{
      "text"    : "the requirement text (grammar fixed)",
      "is_vague": true or false,
      "reason"  : "why it is vague, or empty string if clear"
    }}
  ],
  "was_split"   : true or false,
  "original_count": 1,
  "split_count"   : 2
}}"""

    try:
        response = client.chat.completions.create(
            model       = AGENT_MODEL,
            messages    = [
                {
                    "role"   : "system",
                    "content": (
                        "You are a requirements analyst. "
                        "Split compound requirements and validate each one. "
                        "Respond only in JSON."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens          = 500,
            temperature         = 0,
            response_format     = {"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return result

    except Exception as e:
        print(f"Split/validate error: {e}")
        # Fallback — return original as single non-vague requirement
        return {
            "requirements"   : [{"text": text, "is_vague": False, "reason": ""}],
            "was_split"      : False,
            "original_count" : 1,
            "split_count"    : 1
        }