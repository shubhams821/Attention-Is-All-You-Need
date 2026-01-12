import pandas as pd

train_df = pd.read_csv(r"/content/hin_train.csv", header = None)
test_df = pd.read_csv(r"/content/hin_test.csv", header = None)
valid_df = pd.read_csv(r"/content/hin_valid.csv", header = None)


class vocab:
    def __init__(self, train_df, test_df, valid_df):
        eng_vocab = train_df[0].tolist() + test_df[0].tolist() + valid_df[0].tolist()
        hin_vocab = train_df[1].tolist() + test_df[1].tolist() + valid_df[1].tolist()
        self.max_len = 0
        self.total_char = set()
        for word in eng_vocab:
          chars = list(word)
          self.max_len = max(self.max_len, len(chars))
          for char in chars:
            self.total_char.add(char)
        
        for word in hin_vocab:
          chars = list(word)
          self.max_len = max(self.max_len, len(chars))
          for char in chars:
            self.total_char.add(char)

        self.total_char.add("SOT")
        self.total_char.add("EOD")
        self.total_char.add("PAD")

        self.vocab_for = {tok: i for i, tok in enumerate(self.total_char) }
        self.vocab_back = {i: tok for tok, i in self.vocab_for.items()}

    def tokenize(self, word):
        if word in ["SOT", "EOD", "PAD"]:
          return self.vocab_for[word]
        tokens = list(word)
        return [self.vocab_for[tok] for tok in tokens]


    def detokenize(self, tokens):
        word = ""
        for tok in tokens:
          word += self.vocab_back[tok]
        return word
    

vocablary = vocab(train_df, test_df, valid_df)
