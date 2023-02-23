# coding='utf-8

import requests
import re
import collections
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import asyncio


def get_html(page):
    cookies = {
        '_free_proxy_session': 'BAh7B0kiD3Nlc3Npb25faWQGOgZFVEkiJTg1ZDEzMzUwNDUyMzZkN2UwNzI4YmE0OWQyNmMxZDM3BjsAVEkiEF9jc3JmX3Rva2VuBjsARkkiMUZnbmJlZFNaR1FnbHBVa0h4N3U4NFR3SjkyTUgyL1R5QVA0cXZ5Y1dUcTA9BjsARg%3D%3D--6d4992b5a5c60d2e9a7e0fcffed0ffe89a8b51ed',
        'Hm_lvt_0cf76c77469e965d2957f0553e6ecf59': '1592280847',
        'Hm_lpvt_0cf76c77469e965d2957f0553e6ecf59': '1592281190', }
    headers = {
        'Connection': 'keep-alive',
        'Cache-Control': 'max-age=0',
        'Upgrade-Insecure-Requests': '1',
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
        'Sec-Fetch-Site': 'none',
        'Sec-Fetch-Mode': 'navigate',
        'Sec-Fetch-User': '?1',
        'Sec-Fetch-Dest': 'document',
        'Accept-Language': 'zh-CN,zh;q=0.9,en-US;q=0.8,en;q=0.7',
        'If-None-Match': 'W/"20e6a52e9bd39a291ba761a624c337bd"', }
    url = "{}{}".format('https://www.xicidaili.com/nn/', page)
    response = requests.get(url, headers=headers, cookies=cookies)
    return response


def reLambda(x):
    recom = re.compile(r'[a-z^A-Z\/\n\r\v\<\f\n\r\t\v\=\>\"\        ]')
    newx = re.sub(recom, "", x)
    return newx


def group_adjacent(a, k):
    return list(zip(*([iter(a)] * k)))


def get_data(html):
    return list(map(lambda x: reLambda(x), re.findall(r"<td(.*)</td>", html)+re.findall(r"<td>(.*)</td>", html)))


def get_proxy(betw, page):
    resp = get_html(page)
    print(resp)
    data = get_data(resp.text)
    return group_adjacent(list(filter(lambda x: x != '' and x != "...", data)), betw)


async def main(page):
    return list(filter(lambda x: '高匿' in x, get_proxy(5, page)))

coroutine = main(1)
loop = asyncio.get_event_loop()
task = asyncio.ensure_future(coroutine)
loop.run_until_complete(task)
