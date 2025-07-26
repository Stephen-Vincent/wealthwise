import os

root_dir = "backend"
total_checked = 0
total_fixed = 0
total_skipped = 0

print("🔍 Scanning for incorrect database imports...\n")

for subdir, _, files in os.walk(root_dir):
    for file in files:
        if file.endswith(".py"):
            path = os.path.join(subdir, file)
            total_checked += 1
            try:
                with open(path, "r", encoding="utf-8") as f:
                    content = f.read()

                new_content = content.replace("from database.database", "from database.db")

                if content != new_content:
                    with open(path, "w", encoding="utf-8") as f:
                        f.write(new_content)
                    print(f"✅ Fixed: {path}")
                    total_fixed += 1

            except UnicodeDecodeError:
                print(f"⚠️ Skipped (non-UTF8): {path}")
                total_skipped += 1

print("\n📊 Summary")
print(f"🔎 Total files checked: {total_checked}")
print(f"✅ Total fixed: {total_fixed}")
print(f"⚠️ Total skipped (non-UTF8): {total_skipped}")