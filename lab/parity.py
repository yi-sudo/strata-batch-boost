#!/usr/bin/env python3
"""同時送 N 個 temperature=0 的請求，把輸出存成 JSON，用來比對不同設定的答案是否一致。"""
import json, sys, threading, urllib.request
url, n, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
PROMPTS = ["用 Python 寫一個 LRU cache 類別，附上註解與使用範例。",
           "解釋 TCP 三次握手與四次揮手的流程與原因。",
           "寫一篇 300 字的短文，描述秋天的台北。",
           "計算 1 到 100 之間所有質數的和，逐步說明。"]
res = {}
def go(i):
    body = json.dumps({"model": "x", "max_tokens": 400, "temperature": 0, "reasoning_effort": "none",
                       "messages": [{"role": "user", "content": PROMPTS[i % len(PROMPTS)]}]}).encode()
    r = urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"}), timeout=600)
    res[i] = json.load(r)["choices"][0]["message"]["content"]
th = [threading.Thread(target=go, args=(i,)) for i in range(n)]
[t.start() for t in th]; [t.join() for t in th]
json.dump([res[i] for i in range(n)], open(out, "w"), ensure_ascii=False, indent=1)
print("saved", out)
