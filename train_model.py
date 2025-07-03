pd
import re
from sklearn.model_selection import train_test_split
from langdetect import detect
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
