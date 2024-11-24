import torch
from torch import nn
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from torch.optim import AdamW
from utils.data_loader import generate_data_loader
from models.generator import Generator
from models.discriminator import Discriminator
from models.teacher_model import build_teacher_model
from models.student_model import LSTMStudentModel, GRUStudentModel

def train_ganbert(data_path):
    """
    Trains the full GAN-BERT model for text classification.
    Integrates Teacher, Student, Generator, and Discriminator models.
    
    Args:
        data_path (str): Path to the training dataset (CSV).
    """
    print("Training GAN-BERT Model...")

    # Load the dataset
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    train_loader = generate_data_loader(data_path, tokenizer, max_seq_length=64, batch_size=32)

    # Initialize the Teacher model (pre-trained BERT model)
    teacher_model = build_teacher_model(max_seq_length=64, num_classes=3, learning_rate=5e-5)
    teacher_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Student model (LSTM-based or GRU-based)
    student_model = LSTMStudentModel(input_dim=768, hidden_dim=256, output_dim=3)  # Can switch to GRUStudentModel
    student_model.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the Generator and Discriminator for GAN
    generator = Generator(noise_size=100, output_size=768)  # Output size matches BERT's hidden dimension
    discriminator = Discriminator(input_size=768)  # Matches the size of the input to the generator
    generator.to('cuda' if torch.cuda.is_available() else 'cpu')
    discriminator.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimizers for Teacher, Student, Generator, and Discriminator
    optimizer_teacher = AdamW(teacher_model.parameters(), lr=5e-5)
    optimizer_student = AdamW(student_model.parameters(), lr=5e-5)
    optimizer_generator = AdamW(generator.parameters(), lr=5e-5)
    optimizer_discriminator = AdamW(discriminator.parameters(), lr=5e-5)

    # Loss functions
    criterion = nn.CrossEntropyLoss()

    # Knowledge Distillation Loss function (Kullback-Leibler Divergence)
    def distillation_loss(student_logits, teacher_logits, temperature=3.0):
        student_probs = torch.softmax(student_logits / temperature, dim=-1)
        teacher_probs = torch.softmax(teacher_logits / temperature, dim=-1)
        return nn.KLDivLoss(reduction='batchmean')(torch.log(student_probs), teacher_probs)

    # Adversarial Loss function for Generator and Discriminator
    def adversarial_loss(real_output, fake_output):
        real_loss = torch.mean((real_output - 1) ** 2)  # Fake should be as close to 1 as possible
        fake_loss = torch.mean(fake_output ** 2)  # Fake should be as close to 0 as possible
        return (real_loss + fake_loss) / 2

    # Train the GAN-BERT model
    for epoch in range(3):  # Example: training for 3 epochs
        running_loss_teacher = 0.0
        running_loss_student = 0.0
        running_loss_gan = 0.0

        for batch in train_loader:
            inputs = batch[0].to('cuda' if torch.cuda.is_available() else 'cpu')
            labels = batch[1].to('cuda' if torch.cuda.is_available() else 'cpu')

            # Training Teacher Model (Standard classification loss)
            optimizer_teacher.zero_grad()
            teacher_outputs = teacher_model(inputs, labels=labels)
            teacher_loss = teacher_outputs.loss
            teacher_loss.backward()
            optimizer_teacher.step()
            running_loss_teacher += teacher_loss.item()

            # Generate fake data using the Generator (for GAN)
            noise = torch.randn(inputs.size(0), 100).to('cuda' if torch.cuda.is_available() else 'cpu')
            fake_data = generator(noise)

            # Discriminator: Real vs Fake data
            optimizer_discriminator.zero_grad()
            real_output = discriminator(inputs)
            fake_output = discriminator(fake_data.detach())
            gan_loss = adversarial_loss(real_output, fake_output)
            gan_loss.backward()
            optimizer_discriminator.step()

            # Generator: Minimize adversarial loss (Generate data that the Discriminator thinks is real)
            optimizer_generator.zero_grad()
            fake_output = discriminator(fake_data)
            gen_loss = adversarial_loss(fake_output, fake_output)  # Maximize fake output to 1
            gen_loss.backward()
            optimizer_generator.step()

            # Training Student Model (Knowledge Distillation + Cross-Entropy Loss)
            optimizer_student.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher_model(inputs).logits
            student_logits = student_model(inputs)
            student_loss = distillation_loss(student_logits, teacher_logits) + criterion(student_logits, labels)
            student_loss.backward()
            optimizer_student.step()
            running_loss_student += student_loss.item()

        # Print the losses for each epoch
        print(f"Epoch {epoch+1}:")
        print(f"Teacher Loss: {running_loss_teacher / len(train_loader)}")
        print(f"Student Loss: {running_loss_student / len(train_loader)}")
        print(f"GAN Loss: {running_loss_gan / len(train_loader)}")

    print("GAN-BERT model training completed.")
    teacher_model.save_pretrained('teacher_model')
    student_model.save_pretrained('student_model')
    generator.save_pretrained('generator_model')
    discriminator.save_pretrained('discriminator_model')

