"""Phase 03 — word, character, subword, and pretrained tokenizers."""

from collections import Counter
from pathlib import Path

import sentencepiece as spm
from transformers import AutoTokenizer

GENERATED = Path(__file__).resolve().parent / "generated"
SAMPLE = "Large language models are powerful."
COMPARE_TEXT = """
Artificial intelligence is transforming
software engineering.
"""


def show_bert_tokens():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.tokenize(SAMPLE)
    token_ids = tokenizer.encode(SAMPLE, add_special_tokens=True)
    print("Original text:")
    print(SAMPLE)
    print("\nTokens:")
    print(tokens)
    print("\nToken IDs:")
    print(token_ids)
    print("\nNumber of tokens:")
    print(len(token_ids))
    print("\nDecoded text:")
    print(tokenizer.decode(token_ids))
    return tokenizer


def show_character_pairs():
    text = "low lower lowest"
    words = [list(word) for word in text.split()]
    print("\nCharacter pieces:")
    print(words)
    pairs = Counter()
    for word in words:
        for index in range(len(word) - 1):
            pairs[(word[index], word[index + 1])] += 1
    print("Neighboring character pairs:")
    print(pairs)


def show_character_ids():
    text = "hello"
    vocab = {"h": 0, "e": 1, "l": 2, "o": 3}
    print("\nCharacters:", list(text))
    print("Character IDs:", [vocab[character] for character in text])


def compare_tokenizers():
    print("\nTokenizer comparison")
    for model_name in ("bert-base-uncased", "gpt2"):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        token_ids = tokenizer.encode(COMPARE_TEXT, add_special_tokens=False)
        print("\nModel:", model_name)
        print("Tokens:", tokenizer.tokenize(COMPARE_TEXT))
        print("Number of tokens:", len(token_ids))


def train_sentencepiece():
    GENERATED.mkdir(exist_ok=True)
    corpus_path = GENERATED / "training.txt"
    corpus_path.write_text(
        """
I love artificial intelligence.
Artificial intelligence is changing software engineering.
Large language models process text using tokens.
Tokenization converts text into smaller units.
Machine learning is an important field of computer science.
Natural language processing allows computers to understand text.
Python is widely used for artificial intelligence.
Students are learning about large language models.
""",
        encoding="utf-8",
    )
    model_prefix = str(GENERATED / "my_tokenizer")
    spm.SentencePieceTrainer.train(
        input=str(corpus_path),
        model_prefix=model_prefix,
        vocab_size=50,
        model_type="unigram",
        character_coverage=1.0,
    )
    tokenizer = spm.SentencePieceProcessor(model_file=model_prefix + ".model")
    text = "Artificial intelligence is changing software engineering."
    tokens = tokenizer.encode(text, out_type=str)
    token_ids = tokenizer.encode(text, out_type=int)
    print("\nSentencePiece")
    print("Original text:")
    print(text)
    print("\nTokens:")
    print(tokens)
    print("\nToken IDs:")
    print(token_ids)
    print("\nNumber of tokens:")
    print(len(token_ids))
    print("\nDecoded text:")
    print(tokenizer.decode(token_ids))


def show_special_tokens(tokenizer):
    tokens = tokenizer.tokenize("Hello world")
    token_ids = tokenizer.encode("Hello world", add_special_tokens=True)
    print("\nSpecial tokens")
    print(tokens)
    print(token_ids)
    print("CLS token:", tokenizer.cls_token, tokenizer.cls_token_id)
    print("SEP token:", tokenizer.sep_token, tokenizer.sep_token_id)
    plain_ids = tokenizer.encode(
        "Artificial intelligence is transforming software engineering and many other fields.",
        add_special_tokens=False,
    )
    text = "Artificial intelligence is transforming software engineering and many other fields."
    print("Characters:", len(text))
    print("Tokens:", len(plain_ids))


def whitespace_and_unknown():
    def tokenize(text: str) -> list[str]:
        return text.lower().split()

    text = "Agentic AI systems can use tools"
    tokens = tokenize(text)
    print("\nWhitespace tokens:", tokens)
    print("Token count:", len(tokens))

    vocab = {
        "<UNK>": 0,
        "Large": 1,
        "language": 2,
        "models": 3,
        "understand": 4,
        "text": 5,
    }
    words = "Large language models understand text".split()
    print("Known-word IDs:", [vocab.get(token, vocab["<UNK>"]) for token in words])
    print("Subwords:", tokenizer_subwords())


def tokenizer_subwords():
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    return tokenizer.tokenize("tokenization")


def main():
    print("=== Phase 03 — tokenization ===")
    bert = show_bert_tokens()
    show_character_pairs()
    show_character_ids()
    compare_tokenizers()
    train_sentencepiece()
    show_special_tokens(bert)
    whitespace_and_unknown()
    print("\n[CLS] starts a BERT sequence and [SEP] ends it.")
    print("Subword tokenization splits an uncommon word into known pieces,")
    print("such as token + ##ization, instead of treating the whole word as unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
