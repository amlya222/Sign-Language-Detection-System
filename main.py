import sys
import argparse

def main():
    """
    Main Entry File for Sign Language Detection System
    """
    parser = argparse.ArgumentParser(description="Sign Language Action Recognition System")
    parser.add_argument('--train', action='store_true', help='Kick off the Model Training process (Requires Internet)')
    parser.add_argument('--run', action='store_true', help='Run the real-time GUI application')
    args = parser.parse_args()

    title_banner = """
    ===============================================
       SIGN LANGUAGE REAL-TIME TRANSLATOR SYSTEM  
    ===============================================
    """
    print(title_banner)

    if args.train:
        print("[INFO] Starting Training Pipeline...")
        from model import train_model
        train_model()
    else:
        # Default behavior is to start the GUI
        print("[INFO] Launching Translating Interface...")
        from gui import SignLanguageApp
        import tkinter as tk
        root = tk.Tk()
        app = SignLanguageApp(root)
        root.mainloop()

if __name__ == "__main__":
    main()
