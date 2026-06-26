from transformers import pipeline
import pandas as pd
import json
import subprocess

candidate_labels = [
    "appeal to fear",
    "false dichotomy",
    "loaded language",
    "scapegoating",
    "neutral statement"
]

test_sentences = [
    ("If we don't act now, our children will inherit a ruined world.", "appeal to fear"),
    ("You're either with us or against us — there's no middle ground.", "false dichotomy"),
    ("These corrupt politicians have betrayed every single one of us.", "loaded language"),
    ("It's their fault — they're the reason everything has gone wrong.", "scapegoating"),
    ("The meeting is scheduled for 3pm in the main conference room.", "neutral statement"),
    ("Unless we stop them immediately, everything we love will be destroyed.", "appeal to fear"),
    ("Anyone who disagrees with this policy clearly doesn't care about safety.", "false dichotomy"),
    ("The disgraceful, reckless actions of these so-called leaders shame us all.", "loaded language"),
    ("Immigrants are the reason wages have stagnated for ordinary workers.", "scapegoating"),
    ("The report was published last Tuesday and is available online.", "neutral statement"),
    ("Buy now before this opportunity disappears forever.", "appeal to fear"),
    ("There are only two choices here: total victory or total defeat.", "false dichotomy"),
]

df = pd.DataFrame(test_sentences, columns=["text", "true_label"])
print(df)

print("\nLoading zero-shot classifier (downloads model on first run)...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

predicted_labels = []
confidence_scores = []

for text in df["text"]:
    result = classifier(text, candidate_labels)
    predicted_labels.append(result["labels"][0])  # top predicted label
    confidence_scores.append(result["scores"][0])  # its confidence

df["predicted_label"] = predicted_labels
df["confidence"] = confidence_scores

print("\n=== Classification Results ===")
print(df[["text", "true_label", "predicted_label", "confidence"]].to_string(index=False))

accuracy = (df["true_label"] == df["predicted_label"]).mean()
print(f"\nAccuracy on {len(df)} hand-labeled examples: {accuracy:.2%}")

# Save results
df.to_csv("outputs/nlp_classification_results.csv", index=False)
print("Saved outputs/nlp_classification_results.csv")

# Optional — requires `ollama pull llama3.2` run once in terminal first
'''
def generate_prebunk_explainer(technique):
    prompt = f"""Explain the manipulation technique "{technique}" in 2-3 sentences, 
in a neutral, educational tone, without referencing any specific political topic or claim. 
Focus on how the technique works psychologically, so readers can recognize it in any context."""
    
    result = subprocess.run(
        ["ollama", "run", "llama3.2", prompt],
        capture_output=True, text=True
    )
    return result.stdout.strip()

example_explainer = generate_prebunk_explainer("false dichotomy")
print(f"\nExample pre-bunking explainer:\n{example_explainer}")

with open("outputs/example_prebunk_explainer.txt", "w") as f:
    f.write(example_explainer)
'''