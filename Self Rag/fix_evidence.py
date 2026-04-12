import json

with open('Simple_rag.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_text = "- Evidence: include up to 3 short direct quotes from CONTEXT that support the supported parts."
new_text = "- Evidence: list up to 3 short direct quotes from CONTEXT as plain strings only. Each evidence item MUST be a string, not an object or dict."

fixed = 0
for i, cell in enumerate(nb['cells']):
    new_source = []
    for line in cell['source']:
        if old_text in line:
            new_line = line.replace(old_text, new_text)
            new_source.append(new_line)
            print(f"Cell {i} fixed:")
            print(f"  OLD: {line.strip()}")
            print(f"  NEW: {new_line.strip()}")
            fixed += 1
        else:
            new_source.append(line)
    cell['source'] = new_source

with open('Simple_rag.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print(f"\nTotal fixes: {fixed}")
