import sentencepiece as spm
from pathlib import Path

Path("data").mkdir(exist_ok=True)
with open("data/corpus.txt", "w", encoding="utf-8") as f:
    f.write(
        "سلام. من مانستر چت هستم.\n"
        "Hello. I am Monster Chat.\n"
        "Bonjour. Je suis Monster Chat.\n"
        "Hola. Soy Monster Chat.\n"
        "مرحبا أنا مونستر شات.\n"
    )
spm.SentencePieceTrainer.train(
    input="data/corpus.txt",
    model_prefix="model/tokenizer",
    vocab_size=128,
    model_type="bpe",
    character_coverage=0.9995
)
