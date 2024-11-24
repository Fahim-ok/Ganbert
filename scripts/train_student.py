import torch
import torch.nn as nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from utils.data_loader import generate_data_loader

class LSTMStudentModel(nn.Module):
    """
    LSTM-based Student model for knowledge distillation.
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3, num_layers=2):
        super(LSTMStudentModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, (hn, _) = self.lstm(x)
        out = self.fc(hn[-1])  # Use the last hidden state
        return out

class GRUStudentModel(nn.Module):
    """
    GRU-based Student model for knowledge distillation.
    """
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3, num_layers=2):
        super(GRUStudentModel, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        _, hn = self.gru(x)
        out = self.fc(hn[-1])  # Use the last hidden state
        return out

def train_student(data_path, teacher_model_path, model_type="LSTM"):
    """
    Trains the Student model using Knowledge Distillation from the Teacher model.
    
    Args:
        data_path (str): Path to the training dataset (CSV).
        teacher_model_path (str): Path to the pre-trained Teacher model.
        model_type (str): Type of the student model ('LSTM' or 'GRU').
    """
    print(f"Training Student Model with Knowledge Distillation using {model_type}...")

    # Load the Teacher model
    teacher_model = RobertaForSequenceClassification.from_pretrained(teacher_model_path)
    teacher_model.eval()  # Set teacher model to evaluation mode (no gradient updates)

    # Load the dataset and prepare the DataLoader
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_loader = generate_data_loader(data_path, tokenizer, max_seq_length=64, batch_size=32)

    # Initialize the Student model (LSTM or GRU based on model_type)
    if model_type == "LSTM":
        student_model = LSTMStudentModel(input_dim=768, hidden_dim=256, output_dim=3)
    elif model_type == "GRU":
        student_model = GRUStudentModel(input_dim=768, hidden_dim=256, output_dim=3)
    else:
        print("Invalid model type selected! Please choose 'LSTM' or 'GRU'.")
        return
    
    student_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up the optimizer and loss function
    optimizer = AdamW(student_model.parameters(), lr=5e-5)
    criterion = torch.nn.CrossEntropyLoss()

    # Knowledge Distillation Loss function (Kullback-Leibler Divergence)
    def distillation_loss(student_logits, teacher_logits, temperature=3.0):
        """
        Calculates Knowledge Distillation loss using KL Divergence.
        
        Args:
            student_logits (tensor): The logits from the student model.
            teacher_logits (tensor): The logits from the teacher model.
            temperature (float): The temperature for softening the outputs.
        
        Returns:
            torch.Tensor: The knowledge distillation loss.
        """
        student_probs = torch.softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        return torch.nn.KLDivLoss(reduction='batchmean')(torch.log(student_probs), teacher_probs)

    # Train the Student model with Knowledge Distillation
    student_model.train()
    for epoch in range(3):  # Example: training for 3 epochs
        running_loss = 0.0
        for batch in train_loader:
            # Get the inputs and labels
            inputs = batch[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch[1].to('cuda' if torch.cuda.is_available() else 'cpu')

            optimizer.zero_grad()

            # Get Teacher model predictions
            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)
                teacher_logits = teacher_outputs.logits

            # Get Student model predictions
            student_outputs = student_model(inputs)
            student_logits = student_outputs

            # Calculate the Distillation loss (KL Divergence)
            kd_loss = distillation_loss(student_logits, teacher_logits)

            # Cross-entropy loss
            ce_loss = criterion(student_logits, labels)

            # Total loss is a combination of distillation loss and cross-entropy loss
            loss = ce_loss + kd_loss

            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update the student model parameters

            running_loss += loss.item()

        print(f"Epoch {epoch + 1} - Loss: {running_loss / len(train_loader)}")

    print("Student model training completed.")

    # Save the trained Student model
    student_model.save_pretrained('student_model')

