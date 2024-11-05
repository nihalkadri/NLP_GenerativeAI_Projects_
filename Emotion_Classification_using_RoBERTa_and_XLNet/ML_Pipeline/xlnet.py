import pandas as pd
import ktrain
from ktrain import text

# create a class for xlnet model
class XLNet:

    def __init__(self):
        self.model_name = "xlnet-base-cased"
        self.maxlen = 128
        self.classes = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        self.batch_size = 4                                                    # Reduced the batch_size on purpose
                                                                               # Due to requirement of higher computation power

    def create_transformer(self):
        return text.Transformer(self.model_name, self.maxlen, self.classes, self.batch_size)

