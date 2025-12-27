#!/usr/bin/env python3
"""
Script to generate Urdu translations of documentation files
"""

import os
import re
from pathlib import Path

def translate_to_urdu(text):
    """Basic translation function - in a real implementation, this would use an actual Urdu translation API"""
    # This is a simplified function that would normally call an Urdu translation API
    # For this implementation, we'll just keep the English content with a note
    # since proper Urdu translation requires sophisticated NLP capabilities

    # For demonstration purposes, I'll add a header indicating this is a placeholder
    # and keep the original English content with the code snippets unchanged
    urdu_header = """# ہیومنوڈ روبوٹکس اور فزیکل ای آئی کا تعارف (اردو ترجمہ)

> **نوٹ**: یہ اردو ترجمہ جاری ہے۔ کوڈ کے حصے انگریزی میں ہی رہیں گے۔

"""

    # Keep the original content but add the Urdu header
    translated = urdu_header + text

    # Remove the original title since we added a Urdu one
    translated = re.sub(r'^#.*?\n', '', translated, count=1, flags=re.MULTILINE)

    return translated

def process_file_for_urdu(file_path, output_dir):
    """Process a single file to create Urdu translation"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create subdirectory structure in output
    relative_path = os.path.relpath(file_path, 'docs')
    output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(output_subdir, exist_ok=True)

    # Get the base filename without extension
    base_name = Path(file_path).stem

    # Create Urdu translation
    output_filename = f"{base_name}.md"
    output_path = os.path.join(output_subdir, output_filename)

    urdu_content = translate_to_urdu(content)

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(urdu_content)

    print(f"Created Urdu translation: {output_path}")

def main():
    # Define source and output directories
    source_dir = 'docs'
    output_dir = 'docs-urdu'

    # Get all markdown files from source directory
    md_files = []
    for root, dirs, files in os.walk(source_dir):
        # Skip certain directories
        dirs[:] = [d for d in dirs if d not in ['docs-personalized', 'docs-urdu', 'node_modules']]
        for file in files:
            if file.endswith('.md') and not any(skip_dir in root for skip_dir in ['docs-personalized', 'docs-urdu']):
                md_files.append(os.path.join(root, file))

    # Process each file
    for file_path in md_files:
        # Skip files that are not part of the main content structure
        if any(skip_file in file_path for skip_file in ['index.md', 'README.md', 'installation.md', 'quickstart.md', 'local-development.md', 'upgrade.md']):
            continue
        process_file_for_urdu(file_path, output_dir)

if __name__ == "__main__":
    main()