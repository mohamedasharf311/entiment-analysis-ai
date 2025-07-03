tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Load trained model and tokenizer
model = TFAutoModelForSequenceClassification.from_pretrained("sentiment_model/")
tokenizer = AutoTokenizer.from_pretrained("sentiment_model/")

# Inference function
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    output = model(inputs).logits
    pred = tf.argmax(output, axis=1).numpy()[0]
    labels = {0: "Negative", 1: "Mixed", 2: "Positive"}
    return labels[pred]

# Example
if __name__ == "__main__":
    examples = [
        "Not good at all.",
        "Food was great, service was awful",
        "Wow, great service... not!",
        "happy to be with you"
    ]
    for ex in examples:
        print(f"{ex} → {predict(ex)}")

