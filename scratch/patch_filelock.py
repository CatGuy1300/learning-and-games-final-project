import os

path = "src/utils/tracking.py"
with open(path, "r", encoding="utf-8") as f:
    content = f.read()

# Add import
if "from filelock import FileLock" not in content:
    content = content.replace("import pandas as pd", "import pandas as pd\nfrom filelock import FileLock")

# Wrap the file write
target = """        with open(out_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record) + '\\n')"""
            
replacement = """        lock_path = out_path + ".lock"
        with FileLock(lock_path):
            with open(out_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(record) + '\\n')"""
                
content = content.replace(target, replacement)

with open(path, "w", encoding="utf-8") as f:
    f.write(content)
