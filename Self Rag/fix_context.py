import json

with open('Simple_rag.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

old_issup = """def issup_node(state:State):
    qus =  state['question']
    answer = state['answer']
    context = state['context']

    decision:IsSupDecision = issup_chain.invoke({'question':qus,'answer':answer,'context':context})
    return {
        'issup':decision.issup,
        'evidence':decision.evidence
    }"""

new_issup = """def issup_node(state:State):
    qus =  state['question']
    answer = state['answer']
    context = state.get('context', '')  # safe get

    if not context:
        return {
            'issup': 'no_support',
            'evidence': []
        }

    decision:IsSupDecision = issup_chain.invoke({'question':qus,'answer':answer,'context':context})
    return {
        'issup':decision.issup,
        'evidence':decision.evidence
    }"""

old_gen = """def generation_node(state:State):
    out = genaration_chain.invoke({'question':state['question']})
    return {'answer':out}"""

new_gen = """def generation_node(state:State):
    out = genaration_chain.invoke({'question':state['question']})
    return {'answer': out, 'context': ''}"""

fixed = 0
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])

    if old_issup in src:
        new_src = src.replace(old_issup, new_issup)
        cell['source'] = [line + '\n' for line in new_src.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print("issup_node fixed - Cell " + str(i))
        fixed += 1

    src2 = ''.join(cell['source'])
    if old_gen in src2:
        new_src2 = src2.replace(old_gen, new_gen)
        cell['source'] = [line + '\n' for line in new_src2.split('\n')]
        if cell['source']:
            cell['source'][-1] = cell['source'][-1].rstrip('\n')
        print("generation_node fixed - Cell " + str(i))
        fixed += 1

with open('Simple_rag.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Total fixes: " + str(fixed))
