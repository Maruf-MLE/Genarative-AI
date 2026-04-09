import json

file_path = "d:/Coding/Genarative-AI/test-2.ipynb"
with open(file_path, "r", encoding="utf-8") as f:
    notebook = json.load(f)

for cell in notebook.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    source = cell.get("source", [])
    new_source = []
    
    for line in source:
        # Fix state[docs] to state['docs']
        if "for d in state[docs]:" in line:
            line = line.replace("state[docs]", "state['docs']")
        
        # Add missing fields after refined_context:str
        if "refined_context:str" in line:
            new_source.append(line)
            new_source.append("    good_docs:list[Document]\n")
            new_source.append("    verdict:str\n")
            new_source.append("    reason:str\n")
            new_source.append("    answer:str\n")
            continue
            
        # Fix UPPER_TH to upper_th and LOWER_TH to lower_th
        if "{UPPER_TH}" in line:
            line = line.replace("{UPPER_TH}", "{upper_th}")
        if "{LOWER_TH}" in line:
            line = line.replace("{LOWER_TH}", "{lower_th}")
            
        # Fix list notation
        if "kept_strips:[list[str]]" in line:
            line = line.replace("kept_strips:[list[str]]", "kept_strips:list[str]")
            
        # Fix quotes inside f-strings
        if "f'FAIL :{state['reason']}'" in line:
            line = line.replace("f'FAIL :{state['reason']}'", "f'FAIL: {state[\"reason\"]}'")
        if "f'amviguous:{state['reason']}'" in line:
            line = line.replace("f'amviguous:{state['reason']}'", "f'ambiguous: {state[\"reason\"]}'")
            
        # Fix WEB_search
        if "'WEB_search'" in line:
            line = line.replace("'WEB_search'", "'web_search'")

        new_source.append(line)
        
    cell["source"] = new_source

with open(file_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)
    # the normal ipynb indent is 1, but without trailing newline, juptyer might format it differently, but it's fine.
print("Notebook fixed successfully!")
