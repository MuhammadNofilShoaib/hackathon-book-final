#!/usr/bin/env python3
"""
Script to generate personalized versions of documentation files
"""

import os
import re
from pathlib import Path

def create_personalized_version(content, level):
    """Create a personalized version of content for a specific level"""

    # Split content into sections based on markdown headers
    sections = re.split(r'(^##\s.*?$)', content, flags=re.MULTILINE)

    # Reconstruct with personalized modifications
    personalized_content = []

    for i, section in enumerate(sections):
        if section.startswith('## '):
            # This is a section header
            personalized_content.append(section)
        elif i > 0 and sections[i-1].startswith('## Concept'):
            # This is the content after "## Concept"
            if level == 'beginner':
                # Add more explanations and analogies for beginners
                personalized_content.append(add_beginner_explanations(section))
            elif level == 'intermediate':
                # Add coding tips and concise explanations
                personalized_content.append(add_intermediate_content(section))
            elif level == 'advanced':
                # Add best practices and optimizations
                personalized_content.append(add_advanced_content(section))
            else:
                personalized_content.append(section)
        elif i > 0 and sections[i-1].startswith('## Pseudo-code'):
            # This is the content after "## Pseudo-code"
            if level == 'beginner':
                personalized_content.append(add_beginner_code_explanations(section))
            elif level == 'intermediate':
                personalized_content.append(add_intermediate_code_features(section))
            elif level == 'advanced':
                personalized_content.append(add_advanced_code_optimizations(section))
            else:
                personalized_content.append(section)
        elif i > 0 and sections[i-1].startswith('## Exercises'):
            # This is the content after "## Exercises"
            if level == 'beginner':
                personalized_content.append(add_beginner_exercises(section))
            elif level == 'intermediate':
                personalized_content.append(add_intermediate_exercises(section))
            elif level == 'advanced':
                personalized_content.append(add_advanced_exercises(section))
            else:
                personalized_content.append(section)
        else:
            # Other sections remain mostly the same but with level-specific additions
            personalized_content.append(section)

    # Modify the main title to include level
    result = ''.join(personalized_content)
    if level == 'beginner':
        result = re.sub(r'^(# .*)', r'\1 (Beginner Level)', result, count=1, flags=re.MULTILINE)
    elif level == 'intermediate':
        result = re.sub(r'^(# .*)', r'\1 (Intermediate Level)', result, count=1, flags=re.MULTILINE)
    elif level == 'advanced':
        result = re.sub(r'^(# .*)', r'\1 (Advanced Level)', result, count=1, flags=re.MULTILINE)

    return result

def add_beginner_explanations(content):
    """Add extra explanations and analogies for beginners"""
    # Add more accessible explanations
    additions = "\n\n> **Beginner Tip**: If this concept feels complex, think of it as [simple analogy related to the topic].\n\n"
    return additions + content

def add_intermediate_content(content):
    """Add concise explanations and coding tips for intermediate users"""
    additions = "\n\n> **Coding Tip**: Consider implementing this with [specific technique] for better performance.\n\n"
    return content + additions

def add_advanced_content(content):
    """Add best practices and optimizations for advanced users"""
    additions = "\n\n> **Best Practice**: For production systems, consider [advanced technique] to optimize performance.\n\n> **Performance Note**: This approach has O(n) complexity and may require optimization for large-scale applications.\n\n"
    return content + additions

def add_beginner_code_explanations(content):
    """Add detailed code explanations for beginners"""
    # Add more detailed comments and explanations
    lines = content.split('\n')
    enhanced_lines = []
    for line in lines:
        enhanced_lines.append(line)
        if line.strip().startswith('#') and not 'Beginner Explanation:' in line:
            enhanced_lines.append(f"# Beginner Explanation: {line.strip('# ')}")
    return '\n'.join(enhanced_lines)

def add_intermediate_code_features(content):
    """Add intermediate-level code features"""
    # Add some intermediate concepts
    additions = "\n# Intermediate Implementation Considerations:\n# - Error handling and validation\n# - Performance optimization opportunities\n# - Integration with other systems\n\n"
    return additions + content

def add_advanced_code_optimizations(content):
    """Add advanced code optimizations and techniques"""
    additions = """
# Advanced Implementation:
# - Real-time performance considerations
# - Memory management optimizations
# - Parallel processing opportunities
# - Safety and fault-tolerance measures
# - Hardware-specific optimizations

"""
    return additions + content

def add_beginner_exercises(content):
    """Add beginner-friendly exercises"""
    # Add simpler exercises
    additions = "\n> **Beginner Exercises**: Focus on understanding core concepts and basic implementations.\n\n"
    return additions + content

def add_intermediate_exercises(content):
    """Add intermediate-level exercises"""
    additions = "\n> **Intermediate Exercises**: Emphasize practical implementation and optimization techniques.\n\n"
    return content + additions

def add_advanced_exercises(content):
    """Add advanced exercises"""
    additions = "\n> **Advanced Exercises**: Challenge students with production-level implementations and performance optimization.\n\n"
    return content + additions

def process_file(file_path, output_dir):
    """Process a single file to create personalized versions"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Create subdirectory structure in output
    relative_path = os.path.relpath(file_path, 'docs')
    output_subdir = os.path.join(output_dir, os.path.dirname(relative_path))
    os.makedirs(output_subdir, exist_ok=True)

    # Get the base filename without extension
    base_name = Path(file_path).stem

    # Create personalized versions
    for level in ['beginner', 'intermediate', 'advanced']:
        output_filename = f"{base_name}-{level}.md"
        output_path = os.path.join(output_subdir, output_filename)

        personalized_content = create_personalized_version(content, level)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(personalized_content)

        print(f"Created: {output_path}")

def main():
    # Define source and output directories
    source_dir = 'docs'
    output_dir = 'docs-personalized'

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
        process_file(file_path, output_dir)

if __name__ == "__main__":
    main()