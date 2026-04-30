import json
import re
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

# ── Numbered-list pre-splitter helpers ────────────────────────────────────

def _detect_numbered_list(text):
    """Return True when input has 3+ numbered/lettered/bulleted list markers."""
    pattern = re.compile(r'^\s*(\d+[.)]\s|[a-z][.)]\s|[-•*]\s)', re.MULTILINE)
    return len(pattern.findall(text)) >= 3


def _pre_split_numbered_list(text):
    """
    Parse a numbered/lettered list block into individual requirement dicts.
    Each dict: {'text': str, 'parent': str | None}
    Sub-items carry their parent's text so fragments can be expanded.
    """
    num_re = re.compile(r'^\s*\d+[.)]\s+(.*)')
    let_re = re.compile(r'^\s*[a-z][.)]\s+(.*)')
    bul_re = re.compile(r'^\s*[-•*]\s+(.*)')

    results     = []
    cur_lines   = []     # lines accumulating the current numbered item
    cur_parent  = None   # full text of the current numbered item (parent for sub-items)
    in_subitems = False  # True once the first lettered sub-item has been emitted

    def flush():
        nonlocal cur_lines, in_subitems
        if cur_lines and not in_subitems:
            txt = ' '.join(cur_lines).strip()
            if txt:
                results.append({'text': txt, 'parent': None})
        cur_lines.clear()
        in_subitems = False

    for raw in text.strip().splitlines():
        stripped = raw.strip()
        if not stripped:
            continue

        mn = num_re.match(raw)
        ml = let_re.match(raw)
        mb = bul_re.match(raw)

        if mn:
            flush()
            t           = mn.group(1).strip()
            cur_lines   = [t]
            cur_parent  = t
            in_subitems = False

        elif ml:
            if not in_subitems:
                # Emit the parent numbered item before starting sub-items
                if cur_lines:
                    p = ' '.join(cur_lines).strip()
                    if p:
                        results.append({'text': p, 'parent': None})
                        cur_parent = p
                cur_lines   = []
                in_subitems = True
            results.append({'text': ml.group(1).strip(), 'parent': cur_parent})

        elif mb:
            flush()
            cur_parent = None
            results.append({'text': mb.group(1).strip(), 'parent': None})

        else:
            # Continuation line — append to current item or last sub-item
            if not in_subitems and cur_lines:
                cur_lines.append(stripped)
            elif in_subitems and results:
                results[-1]['text'] += ' ' + stripped

    flush()
    return results


_SUBJECT_RE = re.compile(
    r'^(the\s|a\s|an\s|users?\s|admins?\s|administrators?\s|'
    r'customers?\s|system\s|application\s|app\s|platform\s|service\s|interface\s)',
    re.I,
)
_MODAL_RE = re.compile(r'\b(shall|should|must|will|can|may|need to)\b', re.I)


def _is_fragment(text):
    """Return True if text is likely a sentence fragment (no subject / no modal verb)."""
    if _SUBJECT_RE.match(text):
        return False
    if _MODAL_RE.search(text):
        return False
    return True


def _expand_fragment(fragment, parent):
    """
    Reconstruct a fragment into a complete sentence using the parent requirement.
    Targets the common pattern 'The system shall allow/enable X to [action]'.
    Falls back to 'The system shall [fragment].' for other parents.
    """
    if not fragment:
        return fragment
    frag = fragment.rstrip('.')
    action = frag[0].lower() + frag[1:]

    # Match "... allow/enable/permit/let <actor> to" in parent
    allow_m = re.match(
        r'^(.*?\b(?:allow|enable|permit|let)\s+\w+(?:\s+\w+)?\s+to\s*)',
        parent, re.I,
    )
    if allow_m:
        prefix = allow_m.group(1).rstrip(':').strip() + ' '
        return f"{prefix}{action}."
    return f"The system shall {action}."


# ── Main entry point ───────────────────────────────────────────────────────

def split_and_validate_input(text):
    """
    Analyse stakeholder input and return a structured split result.

    Handles:
      - Single compound sentences  ("The system should be fast and easy to use.")
      - Numbered list blocks        ("1. Login  2. Register  3. View products")
      - Numbered lists with lettered sub-items (a. b. c. …)
      - Mixed FR / NFR sections with fragments and inline compounds
    """
    # ── Step 1: pre-process numbered / lettered list blocks ────────────────
    was_pre_split   = False
    pre_split_texts = []

    if _detect_numbered_list(text):
        raw_items = _pre_split_numbered_list(text)
        for item in raw_items:
            t      = item['text']
            parent = item.get('parent')
            if parent and _is_fragment(t):
                t = _expand_fragment(t, parent)
            if t:
                pre_split_texts.append(t)
        was_pre_split = bool(pre_split_texts)

    # ── Step 2: build the GPT prompt ──────────────────────────────────────
    if was_pre_split:
        items_str = '\n'.join(f'{i + 1}. {t}' for i, t in enumerate(pre_split_texts))
        prompt = f"""You are a requirements engineering expert.

The following {len(pre_split_texts)} requirement statements were extracted from a \
numbered list. Validate and clean each one:

{items_str}

For every item:
1. VALIDATE — Ensure it is a complete, standalone requirement. If it still contains
   multiple requirements joined by "and" or ";", split it into separate items.
2. VAGUENESS — Flag as vague if it lacks measurable criteria, a specific actor,
   or a clear action.
3. GRAMMAR — Fix all typos and errors
   (e.g. "beavaliable" → "be available", "unathorized" → "unauthorized").
4. COMPLETENESS — If a statement is a fragment, rewrite it as a full IEEE 830
   requirement (e.g. "The system shall ...").

Each output item must describe exactly ONE system property or behaviour.
Do NOT group items together. Treat every numbered item as its own requirement.

Respond ONLY in JSON:
{{
  "requirements": [
    {{
      "text"    : "the requirement text (grammar fixed)",
      "is_vague": true or false,
      "reason"  : "why it is vague, or empty string if clear"
    }}
  ],
  "was_split"     : true,
  "original_count": {len(pre_split_texts)},
  "split_count"   : <total count after any further splits>
}}"""
        max_tok = 4000
    else:
        prompt = f"""You are a requirements engineering expert.

Analyze this stakeholder input:
"{text}"

Perform these tasks:
1. SPLIT: If the input contains multiple requirements (joined by "and", ".", ";",
   or other conjunctions), split them into individual requirements.
   Each requirement must describe ONE system property or behaviour only.

   If the input contains a numbered or lettered list, treat each numbered item
   and each lettered sub-item as a separate individual requirement. Do not group
   them. Sub-items that are fragments must be rewritten as complete standalone
   requirement statements.

2. VALIDATE: For each requirement, check if it is:
   - Clear and testable → keep as-is
   - Vague (missing measurable criteria, actor, or specific action) → flag as vague
   - Has grammar issues → fix them
     (e.g. "beavaliable" → "be available", "unathorized" → "unauthorized")

3. Return each requirement as a separate item.

Examples:
- "The system should be fast and easy to use." → two requirements
- "The system shall encrypt passwords. Users must be able to reset their password."
  → two requirements

Respond ONLY in JSON:
{{
  "requirements": [
    {{
      "text"    : "the requirement text (grammar fixed)",
      "is_vague": true or false,
      "reason"  : "why it is vague, or empty string if clear"
    }}
  ],
  "was_split"     : true or false,
  "original_count": 1,
  "split_count"   : <number of requirements>
}}"""
        max_tok = 600

    # ── Step 3: GPT validation / splitting pass ────────────────────────────
    try:
        response = client.chat.completions.create(
            model    = AGENT_MODEL,
            messages = [
                {
                    "role"   : "system",
                    "content": (
                        "You are a requirements analyst. "
                        "Treat every numbered item and every lettered sub-item as a "
                        "separate individual requirement — never group them. "
                        "Respond only in valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens      = max_tok,
            temperature     = 0,
            response_format = {"type": "json_object"},
        )
        result = json.loads(response.choices[0].message.content)

        # Force was_split=True whenever more than one requirement is returned
        if len(result.get('requirements', [])) > 1:
            result['was_split']   = True
            result['split_count'] = len(result['requirements'])

        return result

    except Exception as e:
        print(f"Split/validate error: {e}")
        if was_pre_split and pre_split_texts:
            return {
                "requirements"  : [{"text": t, "is_vague": False, "reason": ""}
                                    for t in pre_split_texts],
                "was_split"     : True,
                "original_count": len(pre_split_texts),
                "split_count"   : len(pre_split_texts),
            }
        return {
            "requirements"  : [{"text": text, "is_vague": False, "reason": ""}],
            "was_split"     : False,
            "original_count": 1,
            "split_count"   : 1,
        }