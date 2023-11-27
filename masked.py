from collections import Counter
from transformers import AutoTokenizer, BertForMaskedLM
import random

# Masked Language Modeling (MLM) modelini yükle


def mask_sentence(sentence, mask_ratio):
    words = sentence.split()
    num_words_to_mask = int(len(words) * mask_ratio)
    masked_indices = random.sample(range(len(words)), num_words_to_mask)

    for index in masked_indices:
        words[index] = '[MASK]'

    return ' '.join(words)


def fill_masked_tokens(model, tokenizer, sentence):
    tokens = tokenizer.tokenize(tokenizer.decode(tokenizer.encode(sentence)))
    masked_indices = [i for i, token in enumerate(tokens) if token == '[MASK]']

    for index in masked_indices:
        input_ids = tokenizer.encode(sentence, return_tensors='pt')
        logits = model(input_ids).logits
        predicted_token_id = logits[0, index].argmax().item()
        predicted_token = tokenizer.decode(predicted_token_id)
        sentence = sentence.replace('[MASK]', predicted_token, 1)

    return sentence


def generate_ensemble_paragraphs(model, tokenizer, original_paragraph, num_paragraphs, mask_ratio):
    ensemble_paragraphs = []

    for _ in range(num_paragraphs):
        masked_paragraph = mask_sentence(original_paragraph, mask_ratio)
        filled_paragraph = fill_masked_tokens(
            model, tokenizer, masked_paragraph)
        ensemble_paragraphs.append(filled_paragraph)

    return ensemble_paragraphs


def create_ensemble_summary(ensemble_paragraphs):
    num_words = len(ensemble_paragraphs[0].split())
    summary = []

    for i in range(num_words):
        word_frequencies = Counter(paragraph.split(
        )[i] for paragraph in ensemble_paragraphs if len(paragraph.split()) > i)
        most_common_word = word_frequencies.most_common(1)[0][0]
        summary.append(most_common_word)

    return ' '.join(summary)


original_paragraph = """
Tiyatro sanatın en eski ve etkileyici formlarından biridir. Sahne üzerinde canlı performansların sergilendiği bu sanat dalı izleyicilere gerçek bir duygusal deneyim sunar. Tiyatro eserleri genellikle bir hikayeyi dramatik bir şekilde anlatarak toplumsal psikolojik veya politik temaları işler.
"""

model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

num_paragraphs = 3
mask_ratio = 0.50

ensemble_paragraphs = generate_ensemble_paragraphs(
    model, tokenizer, original_paragraph, num_paragraphs, mask_ratio)
ensemble_summary = create_ensemble_summary(ensemble_paragraphs)

print("Asıl Metin:")
print(original_paragraph)

print("\nEnsemble Özeti:")
print(ensemble_summary)
