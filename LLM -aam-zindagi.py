class NakliLLM:
    def __init__(self):
        print('LLM created')

    def predict(self,prompt):
        response_list=[
            'Dhaka is the capital of Banglaesh',
            'BPL is a cricket league',
            'AI stands for Artificial intelligence'
        ]
        return {'response': random.choice(response_list)}
    
llm = NakliLLM()
llm.predict('hi')

class template:
    def __init__(self,template,input_variables):
        self.template =template
        self.input_variables=input_variables

    def format(self,input_dict):
        return self.template.format(**input_dict)