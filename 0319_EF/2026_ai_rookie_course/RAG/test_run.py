import os
import requests

def run_lab01_part2():
    print("--- Running Lab 01 Part 2: Data Loading ---")
    os.makedirs('data', exist_ok=True)
    url = "https://www.gutenberg.org/files/11/11-0.txt"
    output_path = "data/alice_in_wonderland.txt"
    
    if not os.path.exists(output_path):
        print(f"Downloading {url}...")
        try:
            response = requests.get(url)
            response.raise_for_status()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(response.text)
            print(f"✅ Successfully downloaded to {output_path}")
        except Exception as e:
            print(f"❌ Download failed: {e}")
    else:
        print(f"📁 File already exists: {output_path}")

    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"📏 File length: {len(content):,} characters")
            print(f"📝 Preview (first 100 chars): {content[:100]}...")

if __name__ == "__main__":
    run_lab01_part2()
