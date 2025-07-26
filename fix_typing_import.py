# fix_typing_import.py
import os
import re

root_dir = "backend"
checked = 0
fixed = 0
skipped = 0

print("🔍 Scanning for invalid 'typing' imports...")

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(subdir, file)
            checked += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Regex to find 'from typing import ...any...' even among others
                new_content = re.sub(r'from typing import ([\w,\s]*\b)any\b', 
                                     lambda m: f"from typing import {m.group(1)}Any", 
                                     content)

                if content != new_content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    fixed += 1
                    print(f"✅ Fixed: {path}")
            except UnicodeDecodeError:
                print(f"⚠️ Skipped (non-UTF8): {path}")
                skipped += 1

print("\n📊 Summary")
print(f"🔎 Total files checked: {checked}")
print(f"✅ Total fixed: {fixed}")
print(f"⚠️ Total skipped (non-UTF8): {skipped}")