# Install required libraries

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from langdetect import detect
import re
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification

# Sample data
data = {
    "text": [
        "I love the product!",
        "Not good at all.",
        "Wow, great service... not!",
        "Food was great, service was awful",
        "C'est fantastique!",
        "Terrible, terrible, terrible."
    ],
    "label": [2, 0, 0, 1, 2, 0]  # 0: negative, 1: mixed, 2: positive
}
df = pd.DataFrame(data)

# Language detection
df['lang'] = df['text'].apply(lambda x: detect(x))

# Clean text
df['cleaned'] = df['text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# Split data into training and validation sets
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['cleaned'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

# Load tokenizer and encode text data
model_name = "cardiffnlp/twitter-xlm-roberta-base-sentiment"  # multilingual & negation-aware
tokenizer = AutoTokenizer.from_pretrained(model_name)

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Create TensorFlow dataset
train_dataset = tf.data.Dataset.from_tensor_slices((
    dict(train_encodings),
    train_labels
)).batch(8)

val_dataset = tf.data.Dataset.from_tensor_slices((
    dict(val_encodings),
    val_labels
)).batch(8)

# Load pre-trained model for sequence classification
num_labels = 3  # negative, mixed, positive
model = TFAutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metrics = [tf.keras.metrics.SparseCategoricalAccuracy('accuracy')]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

# Train the model
model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# Inference function
def predict(text):
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    output = model(inputs).logits
    pred = tf.argmax(output, axis=1).numpy()[0]
    labels = {0: "Negative", 1: "Mixed", 2: "Positive"}
    return labels[pred]

# Example predictions
print(predict("Not good at all."))
print(predict("Food was great, service was awful"))
print(predict("Wow, great service... not!"))
print(predict("happy to be with you"))
