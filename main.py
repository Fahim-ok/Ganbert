import os
import sys
from scripts.train_ganbert import train_ganbert
from scripts.train_teacher import train_teacher
from scripts.train_student import train_student
from scripts.data_analysis import run_analysis

def main():
    print("Welcome to the GAN-BERT Text Classification Project")
    print("Please choose an option:")
    print("1. Train Teacher Model with GAN-BERT")
    print("2. Train Student Model with Knowledge Distillation and GAN-BERT")
    print("3. Train GAN-BERT Model")
    print("4. Perform Data Analysis")
    print("5. Exit")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        # Train Teacher Model with GAN-BERT
        print("\nTraining Teacher Model with GAN-BERT...")
        data_path = input("Enter the path to your dataset (CSV): ")
        if not os.path.exists(data_path):
            print("The specified file path does not exist. Exiting.")
            sys.exit(1)
        train_teacher(data_path)

    elif choice == '2':
        # Train Student Model with Knowledge Distillation and GAN-BERT
        print("\nTraining Student Model with Knowledge Distillation and GAN-BERT...")
        teacher_model_path = input("Enter the path to the trained teacher model: ")
        if not os.path.exists(teacher_model_path):
            print("The specified teacher model path does not exist. Exiting.")
            sys.exit(1)
        data_path = input("Enter the path to your dataset (CSV): ")
        if not os.path.exists(data_path):
            print("The specified file path does not exist. Exiting.")
            sys.exit(1)
        train_student(data_path, teacher_model_path)

    elif choice == '3':
        # Train GAN-BERT Model
        print("\nTraining GAN-BERT Model...")
        data_path = input("Enter the path to your dataset (CSV): ")
        if not os.path.exists(data_path):
            print("The specified file path does not exist. Exiting.")
            sys.exit(1)
        train_ganbert(data_path)

    elif choice == '4':
        # Data Analysis
        print("\nPerforming Data Analysis...")
        data_path = input("Enter the path to your dataset (CSV): ")
        if not os.path.exists(data_path):
            print("The specified file path does not exist. Exiting.")
            sys.exit(1)
        run_analysis(data_path)

    elif choice == '5':
        print("Exiting the program.")
        sys.exit(0)

    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)

if __name__ == "__main__":
    main()
