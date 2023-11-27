from transformers import AutoTokenizer, BertForMaskedLM
import torch


def get_dropout_representation(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        model.eval()  # Dropout katmanını devre dışı bırak
        outputs = model(**inputs).logits
    return outputs.detach().numpy()


def decode_representations(representations, tokenizer):
    decoded_texts = [tokenizer.decode(rep.argmax(axis=1)[0])
                     for rep in representations]
    return decoded_texts


model_name = "ytu-ce-cosmos/turkish-small-bert-uncased"
model = BertForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

text = "Tiyatro sanatın en eski ve etkileyici formlarından biridir. Sahne üzerinde canlı performansların sergilendiği bu sanat dalı izleyicilere gerçek bir duygusal deneyim sunar."

# Temsil alınacak sayı
num_representations = 5

representations = []
for _ in range(num_representations):
    representation = get_dropout_representation(model, tokenizer, text)
    representations.append(representation)

decoded_texts = decode_representations(representations, tokenizer)

print("Decoded Texts:")
for i, decoded_text in enumerate(decoded_texts):
    print(f"Representation {i + 1}: {decoded_text}")
