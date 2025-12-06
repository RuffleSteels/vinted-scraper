import requests

url = f"https://api.telegram.org/bot8552201082:AAEGd7zpkz2yGY8OQkEKpWq2n_yO3LIXqn0/sendMessage"
payload = {"chat_id": '8506286983', "text": 'message'}

requests.post(url, data=payload)
