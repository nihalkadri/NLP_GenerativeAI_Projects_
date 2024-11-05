import pandas as pd
import ktrain
from ktrain import text
import pickle
# create a class for roberta model
class RoBERTa:

    def __init__(self):
        self.model_name = "roberta-base"
        self.maxlen = 512
        self.classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']         
        self.batch_size = 4                                                    # Reduced the batch_size on purpose
                                                                               # Due to requirement of higher computation power
    def create_transformer(self):
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)