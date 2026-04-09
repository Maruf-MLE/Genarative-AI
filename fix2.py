import json

file_path = "d:/Coding/Genarative-AI/test-2.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    src = cell.get("source", [])
    new_src = []
    for line in src:
        # replace the mangled line
        if "refined_context:str" in line and "good_docs:list[Document]" in line:
            new_src.append("    refined_context:str\n")
            new_src.append("    good_docs:list[Document]\n")
            new_src.append("    verdict:str\n")
            new_src.append("    reason:str\n")
            new_src.append("    answer:str\n")
            # If there was a trailing part that wasn't matched properly, it could cause issues.
            # However this line was precisely constructed by my last fix_nb.py so it should be exactly that.
        else:
            new_src.append(line)
    cell["source"] = new_src

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print("Fixed missing newline!")
