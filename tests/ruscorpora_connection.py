import requests
import json

with open('./ruscorpora_api_key.txt') as f:
    api_token = f.readline().strip()


# Проверяем подключение к API НКРЯ
url = "https://ruscorpora.ru/api/v1/auth/check-authenticated/"
headers = {
    'accept': '*/*',
    'Authorization': f'Bearer {api_token}'
}

response = requests.get(url, headers=headers)

print(f'Connection: {response.json()}')
print(response)