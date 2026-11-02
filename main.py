"""
Sign Language Project - Main Menu
"""

import subprocess
import sys

def run(file_name):
    subprocess.run([sys.executable, file_name])

while True:
    print("\n===== SIGN LANGUAGE PROJECT =====")
    print("1. Detect sign from hand gesture")
    print("2. Convert a name into sign language")
    print("3. Collect new training data")
    print("4. Train the model")
    print("5. Exit")
    choice = input("Enter your choice: ")

    if choice == "1":
        run("sign_detector.py")
    elif choice == "2":
        run("name_to_sign.py")
    elif choice == "3":
        run("collect_data.py")
    elif choice == "4":
        run("train_model.py")
    elif choice == "5":
        break
    else:
        print("Invalid choice.")
