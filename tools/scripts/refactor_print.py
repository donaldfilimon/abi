import os
import re

def main():
    base_dir = "tools/cli"
    replaced_count = 0
    file_count = 0

    # Walk through all Zig files in tools/cli
    for root, _, files in os.walk(base_dir):
        for file in files:
            if not file.endswith('.zig'):
                continue
            
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()

            original_content = content
            
            # Simple line-by-line replacement where std.debug.print doesn't span multiple lines in a complex way
            lines = content.split('\n')
            new_lines = []
            file_modified = False
            
            for line in lines:
                if 'std.debug.print(' in line:
                    # check if the file imports utils. If not, we might need to add it, but for now just replace.
                    if 'utils.output.print' not in line:
                        line = line.replace('std.debug.print(', 'utils.output.print(')
                        replaced_count += line.count('utils.output.print(')
                        file_modified = True
                new_lines.append(line)
            
            if file_modified:
                new_content = '\n'.join(new_lines)
                # Ensure utils is imported if we are using it
                if 'utils.output.print' in new_content and 'const utils =' not in new_content:
                    # Not all files have utils imported. We can insert it after std
                    new_content = re.sub(r'(const std = @import("std"));)', r'\1\nconst utils = @import("utils/mod.zig");', new_content)
                
                with open(filepath, 'w') as f:
                    f.write(new_content)
                file_count += 1

    print(f"Replaced {replaced_count} std.debug.print occurrences in {file_count} files.")

if __name__ == "__main__":
    main()
