import os
import re
import subprocess
import sys


def get_changed_lines(file_path):
    abs_file_path = os.path.abspath(file_path)
    file_dir = os.path.dirname(abs_file_path)

    result = subprocess.run(
        ["git", "-C", file_dir, "diff", "-U0", file_path],
        stdout=subprocess.PIPE,
        text=True,
    )

    lines = result.stdout.splitlines()
    line_change_regex = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")

    modified_lines = []

    for line in lines:
        match = line_change_regex.match(line)
        if match:
            start_line = int(match.group(1))
            num_lines = int(match.group(2) or "1")

            # Collect all affected line numbers
            for i in range(num_lines):
                modified_lines.append(start_line + i)

    return modified_lines


if __name__ == "__main__":
    changed_lines = get_changed_lines(sys.argv[1])
    print("Changed lines in the modified file:", changed_lines)
