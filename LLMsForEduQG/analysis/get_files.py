from pathlib import Path

EXCLUDE_DIRS = {"__pycache__", ".git", "node_modules", ".venv", "venv"}
EXCLUDE_FILES = {"combined_output.py"}  # don't include this script itself

def combine_python_files(folder_path, output_filename):
    folder = Path(folder_path)
    output_path = Path(output_filename)

    with output_path.open('w', encoding='utf-8') as outfile:
        for py_file in sorted(folder.rglob('*.py')):  # rglob = recursive
            
            # skip excluded folders
            if any(part in EXCLUDE_DIRS for part in py_file.parts):
                continue
            
            # skip this script itself
            if py_file.name in EXCLUDE_FILES:
                continue

            relative_path = py_file.relative_to(folder)

            outfile.write("=" * 60 + "\n")
            outfile.write(f"FILE: {relative_path}\n")
            outfile.write("=" * 60 + "\n")
            
            with py_file.open('r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n")

    print(f"Done. Output saved to: {output_path}")

if __name__ == "__main__":
    combine_python_files('../LLMsForEduQG', 'combined_output.txt')