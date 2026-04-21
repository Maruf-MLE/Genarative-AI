import requests

def ask_ai(prompt):
    url = "https://carroll-wyoming-cgi-adopt.trycloudflare.com/api/generate"
    payload = {
        "model": "phi3",
        "prompt": prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json().get('response')
        else:
            return f"Error: {response.status_code}"
    except Exception as e:
        return f"Connection Error: {e}"

# ব্যবহার করার নিয়ম
user_input = "how are you "
result = ask_ai(user_input)
print("AI Response:", result)