import json
import sys
from pathlib import Path

def convert_to_fineweb(input_path, output_path=None):
    """
    Convert a plain-text file into FineWeb-Edu JSONL format.
    Each file becomes a single JSON object with key "text".
    """
    input_path = Path(input_path)
    if output_path is None:
        output_path = input_path.with_suffix(".jsonl")

    # Read text
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read().strip()

    # Create FineWeb-Edu entry
    entry = {"text": text}

    # Write to JSONL
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"✅ Saved to {output_path}")
    print(f"📏 Character count: {len(text):,}")
    print(f"💾 Example: {str(entry)[:200]}...")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_to_fineweb.py <input_text_file> [output_jsonl_file]")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    convert_to_fineweb(input_file, output_file)
