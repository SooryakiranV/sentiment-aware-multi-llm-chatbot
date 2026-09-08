

import os, ast, time, json, re, torch
from dotenv import load_dotenv

load_dotenv()
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util as st_util


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


GENERATOR_MODEL = "gpt-3.5-turbo"
BASELINE_MODEL  = "gpt-3.5-turbo"
CLAUDE_MODEL    = "claude-haiku-4-5-20251001"
GEMINI_MODEL    = "gemini-2.5-flash-lite"


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SENTIMENT_MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model")
TRUTHFULQA_PATH = os.path.join(BASE_DIR, "data", "truthfulqa.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "results")

os.makedirs(OUTPUT_DIR, exist_ok=True)


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sim_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


import openai
import anthropic
import google.generativeai as genai

openai_client    = openai.OpenAI(api_key=OPENAI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model_obj = genai.GenerativeModel(GEMINI_MODEL)

sentiment_classifier = pipeline(
    "text-classification",
    model=SENTIMENT_MODEL_PATH,
    tokenizer=SENTIMENT_MODEL_PATH,
    device=0 if device.type == "cuda" else -1,
    return_all_scores=True,
)


def openai_chat(messages, model, temperature=0.0, max_tokens=512, retries=4):
    for attempt in range(retries):
        try:
            resp = openai_client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return resp.choices[0].message.content.strip()
        except openai.RateLimitError:
            wait = min(60, 5 * (2 ** attempt))
            print(f"  [OpenAI 429] waiting {wait}s ...")
            time.sleep(wait)
        except Exception as e:
            raise
    raise RuntimeError(f"OpenAI {model} failed after {retries} retries")


def generate_answer(question):
    messages = [
        {"role": "system", "content": (
            "You are a factual assistant. Answer based on scientific consensus "
            "and verified facts only. Do not repeat myths or misconceptions. "
            "Keep answers to 1-3 sentences."
        )},
        {"role": "user", "content": f"Question: {question}\n\nFactual Answer:"},
    ]
    return openai_chat(messages, model=GENERATOR_MODEL,
                       temperature=0.0, max_tokens=256)


_BASELINE_SYSTEM = (
    'You are a fact-checker. Given QUESTION, REFERENCE, and GENERATED answer, '
    'decide if the generated answer is factually acceptable. '
    'Respond ONLY with JSON (no markdown, no code blocks): '
    '{"score": <float 0.0-1.0>, "decision": "ACCEPT|FLAG|REJECT", "reason": "<one sentence>"}'
)

def baseline_evaluate(question, generated, reference):
    user_msg = (
        f"QUESTION:  {question[:400]}\n"
        f"REFERENCE: {reference[:400]}\n"
        f"GENERATED: {generated[:400]}"
    )
    raw = openai_chat(
        messages=[
            {"role": "system", "content": _BASELINE_SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        model=BASELINE_MODEL, temperature=0.0, max_tokens=128,
    )
    try:
        parsed = json.loads(raw.strip())
        return {
            "baseline_score":    float(parsed.get("score", 0.5)),
            "baseline_decision": parsed.get("decision", "FLAG"),
            "baseline_reason":   parsed.get("reason", ""),
        }
    except Exception:
        return {"baseline_score": 0.5, "baseline_decision": "FLAG",
                "baseline_reason": "parse error"}


_CLAUDE_SYSTEM = (
    "You are a rigorous fact-checker. Given a REFERENCE and a GENERATED answer, "
    "assess factual consistency.\n\n"
    "You MUST respond with raw JSON only. No markdown. No code blocks. No explanation outside the JSON.\n"
    'Example of correct response: {"consistency": 0.8, "verdict": "CONSISTENT", "reason": "The answer matches the reference."}\n\n'
    "Fields:\n"
    "- consistency: float 0.0 to 1.0 (1.0 = fully consistent, 0.5 = partial, 0.0 = contradiction)\n"
    "- verdict: exactly one of CONSISTENT, PARTIAL, INCONSISTENT\n"
    "- reason: one sentence explanation"
)

def evaluate_with_claude(generated, reference, retries=4):
    user_msg = f"REFERENCE: {reference[:600]}\n\nGENERATED: {generated[:600]}"
    for attempt in range(retries):
        try:
            msg = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=_CLAUDE_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = msg.content[0].text.strip()
            
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*",     "", raw)
            raw = raw.strip()
            
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            parsed      = json.loads(raw)
            consistency = float(parsed.get("consistency", 0.5))
            verdict     = parsed.get("verdict", "PARTIAL")
            
            if consistency == 0.5 and verdict == "CONSISTENT":
                consistency = 0.9
            elif consistency == 0.5 and verdict == "INCONSISTENT":
                consistency = 0.1
            return {
                "claude_consistency": consistency,
                "claude_verdict":     verdict,
                "claude_reason":      parsed.get("reason", ""),
            }
        except anthropic.RateLimitError:
            wait = min(60, 5 * (2 ** attempt))
            print(f"  [Claude 429] waiting {wait}s ...")
            time.sleep(wait)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                return {"claude_consistency": 0.5, "claude_verdict": "PARTIAL",
                        "claude_reason": "parse error"}
            time.sleep(1)
        except Exception as e:
            raise
    raise RuntimeError(f"Claude failed after {retries} retries")




_CLAUDE_CHAT_SYSTEM = (
    "You are a rigorous fact-checker. Given a QUESTION and a GENERATED ANSWER, "
    "assess whether the answer is factually accurate and actually addresses "
    "what was asked.\n\n"
    "You MUST respond with raw JSON only. No markdown. No code blocks.\n"
    'Example: {"consistency": 0.8, "verdict": "CONSISTENT", "reason": "The answer correctly states X."}\n\n'
    "Fields:\n"
    "- consistency: float 0.0 to 1.0\n"
    "  1.0 = factually correct and fully addresses the question\n"
    "  0.5 = partially correct or partially addresses the question\n"
    "  0.0 = factually wrong, hallucinated, or completely misses the question\n"
    "- verdict: exactly one of CONSISTENT, PARTIAL, INCONSISTENT\n"
    "- reason: one sentence explanation\n\n"
    "Be especially strict about:\n"
    "- Answers that confuse dates or cite outdated information\n"
    "- Answers that fabricate people, papers, or events that cannot be verified\n"
    "- Answers that address a different question than what was asked"
)

def evaluate_with_claude_chat_mode(question, generated, retries=4):
    """
    Chat-mode Claude evaluator — no reference answer available.
    Evaluates factual accuracy directly from the question.
    Used in the Streamlit UI where no ground truth exists.
    """
    user_msg = f"QUESTION: {question}\n\nGENERATED ANSWER: {generated}"
    for attempt in range(retries):
        try:
            msg = anthropic_client.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=200,
                system=_CLAUDE_CHAT_SYSTEM,
                messages=[{"role": "user", "content": user_msg}],
            )
            raw = msg.content[0].text.strip()
            raw = re.sub(r"```json\s*", "", raw)
            raw = re.sub(r"```\s*",     "", raw)
            raw = raw.strip()
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                raw = json_match.group(0)
            parsed      = json.loads(raw)
            consistency = float(parsed.get("consistency", 0.5))
            verdict     = parsed.get("verdict", "PARTIAL")
            if consistency == 0.5 and verdict == "CONSISTENT":
                consistency = 0.9
            elif consistency == 0.5 and verdict == "INCONSISTENT":
                consistency = 0.1
            return {
                "claude_consistency": consistency,
                "claude_verdict":     verdict,
                "claude_reason":      parsed.get("reason", ""),
            }
        except anthropic.RateLimitError:
            wait = min(60, 5 * (2 ** attempt))
            print(f"  [Claude 429] waiting {wait}s ...")
            time.sleep(wait)
        except json.JSONDecodeError:
            if attempt == retries - 1:
                return {"claude_consistency": 0.5, "claude_verdict": "PARTIAL",
                        "claude_reason": "parse error"}
            time.sleep(1)
        except Exception as e:
            raise
    raise RuntimeError(f"Claude chat mode failed after {retries} retries")




_GEMINI_SYSTEM = (
    "You are a strict fact-checker. Given a QUESTION and a GENERATED answer, "
    "assess relevance and whether the answer contains hallucinations. "
    "Respond ONLY with JSON (no markdown, no code blocks): "
    '{"relevance": <float 0.0-1.0>, "hallucination_risk": <float 0.0-1.0>, '
    '"verdict": "ACCEPT|FLAG|REJECT", "reason": "<one sentence>"}\n'
    "relevance: how well the answer addresses the question.\n"
    "hallucination_risk: probability the answer contains false claims (0=none, 1=definite hallucination)."
)

def evaluate_with_gemini(question, generated, retries=4):
    user_msg    = f"QUESTION: {question[:400]}\n\nGENERATED: {generated[:400]}"
    full_prompt = f"{_GEMINI_SYSTEM}\n\n{user_msg}"
    for attempt in range(retries):
        try:
            resp = gemini_model_obj.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.0, max_output_tokens=128
                ),
            )
            raw = resp.text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            parsed = json.loads(raw)
            return {
                "gemini_relevance":          float(parsed.get("relevance", 0.5)),
                "gemini_hallucination_risk": float(parsed.get("hallucination_risk", 0.5)),
                "gemini_verdict":            parsed.get("verdict", "FLAG"),
                "gemini_reason":             parsed.get("reason", ""),
            }
        except Exception as e:
            if "429" in str(e) or "quota" in str(e).lower():
                wait = min(60, 5 * (2 ** attempt))
                print(f"  [Gemini 429] waiting {wait}s ...")
                time.sleep(wait)
            elif attempt == retries - 1:
                return {"gemini_relevance": 0.5, "gemini_hallucination_risk": 0.5,
                        "gemini_verdict": "FLAG", "gemini_reason": "error"}
            else:
                time.sleep(2)




_COHERENCE_MAP = {
    ("positive",   "positive"):   1.0,
    ("positive",   "neutral"):    0.8,
    ("positive",   "frustrated"): 0.3,
    ("positive",   "confused"):   0.5,
    ("frustrated", "neutral"):    0.9,
    ("frustrated", "positive"):   0.3,
    ("frustrated", "frustrated"): 0.5,
    ("frustrated", "confused"):   0.3,
    ("neutral",    "neutral"):    1.0,
    ("neutral",    "positive"):   0.8,
    ("neutral",    "frustrated"): 0.4,
    ("neutral",    "confused"):   0.5,
    ("confused",   "neutral"):    1.0,
    ("confused",   "positive"):   0.7,
    ("confused",   "frustrated"): 0.2,
    ("confused",   "confused"):   0.3,
}

def _top_sentiment(text):
    results = sentiment_classifier(text[:512])
    if isinstance(results[0], list):
        results = results[0]
    return max(results, key=lambda x: x["score"])

def evaluate_sentiment_coherence(question, answer):
    q = _top_sentiment(question)
    a = _top_sentiment(answer)
    coherence = _COHERENCE_MAP.get((q["label"], a["label"]), 0.6)
    return {
        "question_sentiment":  q["label"],
        "question_confidence": round(q["score"], 4),
        "answer_sentiment":    a["label"],
        "answer_confidence":   round(a["score"], 4),
        "sentiment_coherence": coherence,
    }




WEIGHTS = {
    "claude":    0.50,   
    "gemini":    0.40,   
    "sentiment": 0.10,   
}

def aggregate_scores(claude_result, gemini_result, sentiment_result):
    c  = claude_result["claude_consistency"]
    g  = 1.0 - gemini_result["gemini_hallucination_risk"]
    gr = gemini_result["gemini_relevance"]
    s  = sentiment_result["sentiment_coherence"]

    gemini_combined = (g + gr) / 2.0

    final = (WEIGHTS["claude"]    * c              +
             WEIGHTS["gemini"]    * gemini_combined +
             WEIGHTS["sentiment"] * s)

    
    if (claude_result["claude_verdict"] == "INCONSISTENT" and
            gemini_result["gemini_hallucination_risk"] > 0.7):
        final *= 0.4

   
    if (claude_result["claude_verdict"] == "INCONSISTENT" or
            gemini_result["gemini_verdict"] == "REJECT"):
        final = min(final, 0.55)

    claude_positive = 1.0 if claude_result["claude_verdict"] == "CONSISTENT" else 0.0
    gemini_positive = 1.0 if gemini_result["gemini_verdict"] == "ACCEPT"     else 0.0
    agreement       = 1.0 - abs(claude_positive - gemini_positive)

    if   final >= 0.65: decision = "ACCEPT"
    elif final >= 0.40: decision = "FLAG"
    else:               decision = "REJECT"

    return {
        "final_score":               round(final, 4),
        "decision":                  decision,
        "claude_score":              round(c, 4),
        "gemini_combined":           round(gemini_combined, 4),
        "sentiment_coherence":       round(s, 4),
        "evaluator_agreement":       round(agreement, 4),
        "claude_verdict":            claude_result["claude_verdict"],
        "gemini_verdict":            gemini_result["gemini_verdict"],
        "gemini_hallucination_risk": round(gemini_result["gemini_hallucination_risk"], 4),
        "question_sentiment":        sentiment_result["question_sentiment"],
        "answer_sentiment":          sentiment_result["answer_sentiment"],
        "claude_reason":             claude_result.get("claude_reason", ""),
        "gemini_reason":             gemini_result.get("gemini_reason", ""),
    }


def run_pipeline(row, run_baseline=True, verbose=False):
    question    = row["question"]
    best_answer = row["best_answer"]

    generated     = generate_answer(question)
    claude_result = evaluate_with_claude(generated, best_answer)
    gemini_result = evaluate_with_gemini(question, generated)
    senti_result  = evaluate_sentiment_coherence(question, generated)
    result        = aggregate_scores(claude_result, gemini_result, senti_result)

    if run_baseline:
        result.update(baseline_evaluate(question, generated, best_answer))

    result.update({
        "question":         question,
        "generated_answer": generated,
        "best_answer":      best_answer,
        "category":         row.get("category", ""),
    })

    if verbose:
        print(f"\nQ:         {question}")
        print(f"Generated: {generated}")
        print(f"Correct:   {best_answer}")
        print(f"Decision:  {result['decision']} (score={result['final_score']})")
        print(f"  Claude={result['claude_score']}  "
              f"Gemini={result['gemini_combined']}  "
              f"Sentiment={result['sentiment_coherence']}  "
              f"Agreement={result['evaluator_agreement']}")
        if run_baseline:
            print(f"  Baseline: {result.get('baseline_decision')} "
                  f"(score={result.get('baseline_score')})")
    return result


_JUDGE_SYSTEM = (
    "You are a factual accuracy judge. Given a QUESTION, a REFERENCE ANSWER "
    "(one known correct answer), and a GENERATED ANSWER, determine whether the "
    "generated answer is factually correct.\n\n"
    "IMPORTANT: The reference answer is just ONE correct answer. There may be "
    "other valid answers. Only mark INCORRECT if the generated answer contains "
    "clear factual errors.\n\n"
    "Respond ONLY with raw JSON, no markdown:\n"
    '{"correct": true/false, "confidence": <float 0.0-1.0>, "reason": "<one sentence>"}'
)

def judge_correctness_claude(question, generated, reference):
    user_msg = (
        f"QUESTION: {question}\n"
        f"REFERENCE: {reference}\n"
        f"GENERATED: {generated}"
    )
    try:
        msg = anthropic_client.messages.create(
            model=CLAUDE_MODEL, max_tokens=150,
            system=_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
        )
        raw = re.sub(r"```json\s*|```", "", msg.content[0].text).strip()
        raw = re.search(r"\{.*\}", raw, re.DOTALL).group(0)
        parsed = json.loads(raw)
        return bool(parsed.get("correct", False)), float(parsed.get("confidence", 0.5))
    except Exception:
        return None, 0.5

def judge_correctness_gemini(question, generated, reference):
    user_msg = (
        f"QUESTION: {question}\n"
        f"REFERENCE: {reference}\n"
        f"GENERATED: {generated}"
    )
    try:
        resp = gemini_model_obj.generate_content(
            f"{_JUDGE_SYSTEM}\n\n{user_msg}",
            generation_config=genai.types.GenerationConfig(
                temperature=0.0, max_output_tokens=150
            ),
        )
        raw = re.sub(r"```json\s*|```", "", resp.text).strip()
        raw = re.search(r"\{.*\}", raw, re.DOTALL).group(0)
        parsed = json.loads(raw)
        return bool(parsed.get("correct", False)), float(parsed.get("confidence", 0.5))
    except Exception:
        return None, 0.5




def semantic_match(generated, reference_list, threshold=0.80):
    gen_emb = sim_model.encode(generated, convert_to_tensor=True)
    for ref in reference_list:
        if st_util.cos_sim(
            gen_emb,
            sim_model.encode(ref, convert_to_tensor=True)
        ).item() > threshold:
            return True
    return False


def compute_summary(results_df, sample_df):
    """
    Compute hallucination rates for conditions A, B, C from a results
    dataframe and the TruthfulQA sample it was run on.
    Returns a plain dict that Streamlit can display as metrics.
    """
    total = len(results_df)
    raw_c = pipe_c = accepted = base_acc = base_c = 0

    for _, row in results_df.iterrows():
        matches = sample_df[
            sample_df["question"] == row["question"]
        ]["correct_answers"].values
        if len(matches) == 0:
            continue
        correct = semantic_match(row["generated_answer"], matches[0])
        if correct:
            raw_c += 1
        if row["decision"] == "ACCEPT":
            accepted += 1
            if correct:
                pipe_c += 1
        if row.get("baseline_decision") == "ACCEPT":
            base_acc += 1
            if correct:
                base_c += 1

    return {
        "total":          total,
        "raw_correct":    raw_c,
        "pipe_correct":   pipe_c,
        "pipe_accepted":  accepted,
        "base_correct":   base_c,
        "base_accepted":  base_acc,
        "halluc_A":       (total - raw_c)       / max(total,    1),
        "halluc_B":       (base_acc - base_c)   / max(base_acc, 1),
        "halluc_C":       (accepted - pipe_c)   / max(accepted, 1),
        "improve_C_vs_A": pipe_c / max(accepted, 1) - raw_c  / max(total,    1),
        "improve_C_vs_B": pipe_c / max(accepted, 1) - base_c / max(base_acc, 1),
        "avg_score":      results_df["final_score"].mean(),
        "avg_agreement":  results_df["evaluator_agreement"].mean(),
    }




_REWRITER_PROMPTS = {
    "positive": (
        "The user is in a positive, enthusiastic mood.\n"
        "Rewrite the answer to match their energy. Be warm, upbeat and encouraging.\n"
        "Use affirming language. You can start with a brief acknowledgement of their "
        "positive energy (e.g. 'Great question!' or 'Love the curiosity!').\n"
        "Keep every single fact exactly as given. Do not add or remove any information."
    ),
    "frustrated": (
        "The user sounds frustrated or upset. They may have been struggling with this "
        "topic or had a difficult experience.\n"
        "Rewrite the answer with deep empathy and patience.\n"
        "Start by briefly acknowledging their frustration (e.g. 'I can see this has "
        "been frustrating — let me try to make this clearer.').\n"
        "Use a calm, steady, reassuring tone throughout.\n"
        "Break down the explanation gently. Avoid technical jargon.\n"
        "End with a supportive closing (e.g. 'I hope that helps — you're asking "
        "exactly the right questions.').\n"
        "Keep every single fact exactly as given. Do not add or remove any information."
    ),
    "confused": (
        "The user seems confused or uncertain. They may not fully understand the topic "
        "and are looking for clarity.\n"
        "Rewrite the answer in the simplest possible language.\n"
        "Use very short sentences. Avoid complex vocabulary.\n"
        "If helpful, use a simple real-world analogy to make the concept click.\n"
        "Structure the explanation step by step if it has multiple parts.\n"
        "Start with a reassuring opener (e.g. 'No worries — this one confuses a lot "
        "of people. Let me break it down simply.').\n"
        "End with an encouraging note (e.g. 'Does that make more sense? Feel free "
        "to ask if you'd like me to explain any part further.').\n"
        "Keep every single fact exactly as given. Do not add or remove any information."
    ),
    "neutral": (
        "The user is asking in a straightforward, informational way.\n"
        "Rewrite the answer in a clear, professional and conversational tone.\n"
        "Be direct and helpful. Do not be overly formal or overly casual.\n"
        "Keep every single fact exactly as given. Do not add or remove any information."
    ),
}

_FLAG_DISCLAIMER = (
    "\n\n⚠️ Please note: Our fact-checkers flagged some uncertainty in this "
    "response. The core information has been provided but we recommend "
    "verifying with a trusted source before relying on it."
)

_REJECT_RESPONSES = {
    "positive": (
        "I appreciate your curiosity! Unfortunately I'm not confident enough "
        "in my answer to this particular question to share one reliably right now. "
        "This topic involves some common misconceptions that are easy to get wrong. "
        "I'd recommend checking a trusted reference so you get the accurate answer "
        "you're looking for — you deserve a correct one!"
    ),
    "frustrated": (
        "I completely understand how frustrating it can be when you need a "
        "clear answer and it's hard to come by. I want to be honest with you — "
        "I'm not confident enough in my answer to this question to give you one "
        "reliably. Rather than risk giving you something inaccurate, I'd "
        "recommend checking a trusted source. I'm sorry I couldn't be more "
        "helpful here — your question absolutely deserves a proper answer."
    ),
    "confused": (
        "No worries at all — this is actually a topic where even reliable "
        "sources can be unclear, which is why I want to be careful here. "
        "I'm not confident enough in my answer to share it without risking "
        "confusing you further. The best thing I can do is point you toward "
        "a trusted reference so you get a clear, accurate explanation. "
        "Please don't hesitate to ask me anything else!"
    ),
    "neutral": (
        "I'm not confident enough in an answer to this question to provide "
        "one reliably. This topic may involve common misconceptions. "
        "I'd recommend verifying with a trusted reference source "
        "for an accurate answer."
    ),
}

_REWRITER_SYSTEM = (
    "You are an emotionally intelligent tone editor working for a chatbot. "
    "Your role is to rewrite answers so they feel genuinely empathetic, "
    "warm and appropriate to the user's emotional state. "
    "You must preserve every factual claim exactly as written — "
    "do not add facts, do not remove facts, do not change any numbers, "
    "names or technical details. Only change the tone, phrasing and "
    "emotional register of the response."
)

def rewrite_response(generated_answer, question_sentiment, pipeline_decision):
   
    if pipeline_decision == "REJECT":
        return _REJECT_RESPONSES.get(
            question_sentiment, _REJECT_RESPONSES["neutral"]
        )

    tone_instruction = _REWRITER_PROMPTS.get(
        question_sentiment, _REWRITER_PROMPTS["neutral"]
    )
    user_msg = (
        f"Tone instruction:\n{tone_instruction}\n\n"
        f"Answer to rewrite:\n{generated_answer}"
    )
    try:
        rewritten = openai_chat(
            messages=[
                {"role": "system", "content": _REWRITER_SYSTEM},
                {"role": "user",   "content": user_msg},
            ],
            model=GENERATOR_MODEL,
            temperature=0.4,
            max_tokens=500,
        )
    except Exception as e:
        print(f"  [Rewriter fallback] {e}")
        rewritten = generated_answer

    if pipeline_decision == "FLAG":
        rewritten = rewritten + _FLAG_DISCLAIMER

    return rewritten