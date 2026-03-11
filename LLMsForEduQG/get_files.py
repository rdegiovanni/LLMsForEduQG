from pathlib import Path


def combine_python_files(folder_path, output_filename):
    folder = Path(folder_path)
    output_path = Path(output_filename)

    with output_path.open('w', encoding='utf-8') as outfile:
        for py_file in folder.glob('*.py'):
            
            outfile.write("-----\n")
            outfile.write(f"{py_file.name}\n")
            outfile.write("-----\n")
            
            with py_file.open('r', encoding='utf-8') as infile:
                outfile.write(infile.read())
                outfile.write("\n\n") 

# Example usage:
if __name__ == "__main__":
    combine_python_files('../LLMsForEduQG', 'combined_output.txt')