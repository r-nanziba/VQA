import matplotlib.pyplot as plt

epochs = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
roberta_train = [2.1989, 1.0954, 0.8723, 0.8306, 0.7460, 0.7251, 0.7128, 0.6500, 0.6606, 0.6229]
roberta_eval  = [1.2204, 1.0202, 0.9168, 0.8817, 0.8927, 0.8741, 0.8612, 0.8524, 0.8598, 0.8415]
bert_train    = [2.6459, 1.2551, 1.0267, 0.9098, 0.8185, 0.7987, 0.7908, 0.7069, 0.7366, 0.6857]
bert_eval     = [1.4106, 1.1627, 0.9770, 0.9409, 0.9025, 0.8788, 0.8537, 0.8798, 0.8610, 0.8604]

plt.figure(figsize=(10, 6))
plt.plot(epochs, roberta_train, label='RoBERTa+BEiT Train Loss', marker='o', color='blue')
plt.plot(epochs, roberta_eval,  label='RoBERTa+BEiT Eval Loss',  marker='o', linestyle='--', color='blue')
plt.plot(epochs, bert_train,    label='BERT+ViT Train Loss',      marker='s', color='red')
plt.plot(epochs, bert_eval,     label='BERT+ViT Eval Loss',       marker='s', linestyle='--', color='red')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Evaluation Loss Curves')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print('Saved to loss_curve.png')