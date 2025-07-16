import os
import shutil
from agent.gaia_agent import GAIAAgent
import pandas as pd
from PIL import Image
import wave
from pptx import Presentation
import zipfile
import json

TEST_DIR = "GAIA/2023/validation"

def create_dummy_files():
    os.makedirs(TEST_DIR, exist_ok=True)

    # .txt
    with open(os.path.join(TEST_DIR, "example.txt"), "w") as f:
        f.write("This is a dummy text file for GAIA.")

    # .csv
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    df.to_csv(os.path.join(TEST_DIR, "example.csv"), index=False)

    # .xlsx
    df.to_excel(os.path.join(TEST_DIR, "example.xlsx"), index=False)

    # .pdb
    with open(os.path.join(TEST_DIR, "example.pdb"), "w") as f:
        f.write("HEADER    DUMMY PDB FILE\nATOM      1  N   ALA A   1")

    # .pdf
    with open(os.path.join(TEST_DIR, "example.pdf"), "wb") as f:
        f.write(b"%PDF-1.4\n%Dummy PDF")

    # .jpg
    img = Image.new("RGB", (100, 100), color="blue")
    img.save(os.path.join(TEST_DIR, "example.jpg"))

    # .png
    img.save(os.path.join(TEST_DIR, "example.png"))

    # .wav
    wav_path = os.path.join(TEST_DIR, "example.wav")
    with wave.open(wav_path, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b'\x00\x00' * 44100)

    # .pptx
    ppt = Presentation()
    slide_layout = ppt.slide_layouts[0]
    slide = ppt.slides.add_slide(slide_layout)
    slide.shapes.title.text = "Dummy PPTX Slide"
    ppt.save(os.path.join(TEST_DIR, "example.pptx"))

    # .zip
    zip_path = os.path.join(TEST_DIR, "example.zip")
    with zipfile.ZipFile(zip_path, 'w') as zipf:
        zipf.writestr("dummy.txt", "This is a file inside a zip archive.")

    # .jsonld
    jsonld_data = {
        "@context": "http://schema.org",
        "@type": "Person",
        "name": "Dummy Name",
        "jobTitle": "Software Agent"
    }
    with open(os.path.join(TEST_DIR, "example.jsonld"), "w") as f:
        json.dump(jsonld_data, f, indent=2)

    # .py
    with open(os.path.join(TEST_DIR, "example.py"), "w") as f:
        f.write("print('Hello from dummy Python script')\n")

    print("✅ All dummy files created successfully!")

def test_file_loading():
    agent = GAIAAgent(gaia_files_dir=TEST_DIR)
    test_files = [
        "example.txt",
        "example.csv",
        "example.xlsx",
        "example.pdb",
        "example.pdf",
        "example.jpg",
        "example.png",
        "example.wav",
        "example.pptx",
        "example.zip",
        "example.jsonld",
        "example.py",
        "example.docx"
    ]

    print(f"\n🔍 [TEST] Testing GAIA Agent File Loading...\n")

    for file_name in test_files:
        print(f"\n--- Testing file: {file_name} ---")
        result = agent._load_file_content(file_name)
        if result is None:
            print("❌ Failed to load or unsupported file format.")
        else:
            print("✅ File loaded successfully. Preview:")
            print(result[:500])

def clean_up():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
        print(f"\n🧹 Cleaned up test files from: {TEST_DIR}")

if __name__ == "__main__":
    create_dummy_files()
    test_file_loading()
    clean_up()
