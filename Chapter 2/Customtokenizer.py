import re

class VocabBuilder:
    def __init__(self, text):
        self.tokens = self._tokenize(text)
        self.vocab = self._build_vocab()
    
    def _tokenize(self, text):
        tokens = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        return [item.strip() for item in tokens if item.strip()]
    
    def _build_vocab(self):
        all_words = sorted(set(self.tokens))
        all_words.extend(["<|endoftext|>", "<|unk|>"])
        return {token: i for i, token in enumerate(all_words)}
    
    def get_vocab(self):
        return self.vocab
    
    def vocab_size(self):
        return len(self.vocab)

class SimpleTokenizerV1:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    def encode(self, text):
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

class SimpleTokenizerV2:
    def __init__(self, vocab):
        self.str_to_int = vocab
        self.int_to_str = { i:s for s,i in vocab.items()}
        
    def encode(self, text):
        preprocessed = re.split(r'(<\|endoftext\|>|[,.?_!"()\']|--|\s)', text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in self.str_to_int else "<|unk|>" for item in preprocessed]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text

with open('The_Verdict.txt',"r",encoding='utf-8') as f:
    raw_data = f.read()

vocab1 = VocabBuilder(raw_data)
vocab = vocab1.get_vocab()

tokenizer = SimpleTokenizerV2(vocab)
t1 = "Hello, do you like tea?"
t2 = "In the sunlit terraces of the palace."
text = "<|endoftext|>".join((t1,t2))
tid = tokenizer.encode(text)
print(tid)
print(tokenizer.decode(tid))