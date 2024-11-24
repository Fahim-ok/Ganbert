from transformers import RobertaForSequenceClassification, RobertaTokenizer
import torch
from torch.utils.data import DataLoader
from utils.data_loader import generate_data_loader

def train_teacher(data_path):
    print("Training Teacher Model...")

    # Load the dataset
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_loader = generate_data_loader(data_path, tokenizer, max_seq_length=64, batch_size=32)

    # Define Teacher Model (BERT)
    model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)  # Example for 3 classes
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Train the Teacher Model
    model.train()
    for epoch in range(3):  # Example: 3 epochs
        for batch in train_loader:
            inputs = batch[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch[1].to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()
            outputs = model(inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
        
        print(f"Epoch {epoch+1} completed")

    print("Teacher model training completed.")
    model.save_pretrained('teacher_model')
