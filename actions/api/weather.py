

import requests


def weather_api(city):
    # url = 'https://tianqiapi.com/api?unescape=1&version=v1&appid=37888876&appsecret=cRwpG1rw'
    url = 'https://tianqiapi.com/api?unescape=1&version=v1&appid=43188428&appsecret=KSeORh7Z '
    params = {}
    params['city'] = city
    r = requests.request("GET", url, params=params)
    x = r.content
    data = eval(x)
    return data['data'], data['city']



