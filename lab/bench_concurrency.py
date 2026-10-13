#!/usr/bin/env python3
"""對正在跑的 Strata 同時丟 N 個請求，量單請求 vs 並發的 tokens/s。

每輪：N 個 client 同時送出（每個 prompt 不同，避免共用 prefix cache），串流接收，記錄
  - TTFT：送出到第一個 token
  - 單請求速度：output_tokens / (結束 - 第一個 token)
  - 總吞吐：所有 output_tokens / (最後一個結束 - 第一個送出)
只用標準函式庫，不需額外套件。
"""
import argparse, json, statistics, threading, time, urllib.request

TOPICS = ["一座海邊的燈塔", "火星上的第一間咖啡館", "會說話的老鐘錶", "深山裡的郵差", "被遺忘的圖書館",
          "沙漠裡的雨季", "地鐵末班車", "極地研究站的冬天", "古城的夜市", "太空電梯的維修工",
          "森林裡的天文台", "海底城市的學校", "雲端上的農場", "失眠的機器人", "島上的最後一座風車", "時間旅行的郵局"]


def one_request(url, model, idx, max_tokens, out, start_evt):
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens, "temperature": 0.7,
        "reasoning_effort": "none",
        "messages": [{"role": "user", "content":
                      f"請寫一篇至少 1500 字的繁體中文短篇小說，主題是「{TOPICS[idx % len(TOPICS)]}」（編號 {idx}）。"
                      "要有完整情節與大量細節描寫，不要分點，不要提前結束。"}],
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start_evt.wait()
    t0 = time.perf_counter(); t_first = None; n_chunks = 0; usage = None; err = None
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    usage = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        if t_first is None:
                            t_first = time.perf_counter()
                        n_chunks += 1
    except Exception as e:  # noqa: BLE001
        err = str(e)
    t1 = time.perf_counter()
    toks = (usage or {}).get("completion_tokens") or n_chunks
    out[idx] = {"t0": t0, "t_first": t_first, "t1": t1, "tokens": toks, "error": err}


def run_round(url, model, clients, max_tokens):
    out = {}; start_evt = threading.Event()
    th = [threading.Thread(target=one_request, args=(url, model, i, max_tokens, out, start_evt)) for i in range(clients)]
    for t in th: t.start()
    time.sleep(0.2); start_evt.set()
    for t in th: t.join()
    rs = [out[i] for i in range(clients)]
    ok = [r for r in rs if not r["error"] and r["t_first"]]
    if not ok:
        return {"clients": clients, "errors": [r["error"] for r in rs]}
    wall = max(r["t1"] for r in ok) - min(r["t0"] for r in ok)
    per = [r["tokens"] / max(r["t1"] - r["t_first"], 1e-6) for r in ok]
    ttft = [r["t_first"] - r["t0"] for r in ok]
    return {"clients": clients, "ok": len(ok), "errors": [r["error"] for r in rs if r["error"]],
            "tokens": sum(r["tokens"] for r in ok), "wall_s": wall,
            "aggregate_tok_s": sum(r["tokens"] for r in ok) / wall,
            "per_req_tok_s_mean": statistics.mean(per), "per_req_tok_s_min": min(per),
            "ttft_mean_s": statistics.mean(ttft), "ttft_max_s": max(ttft)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    ap.add_argument("--clients", default="1,2,4,8")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--label", default="")
    ap.add_argument("--json-out")
    a = ap.parse_args()
    url = a.url.rstrip("/") + "/v1/chat/completions"
    run_round(url, a.model, 1, 64)  # 暖機，不計分
    results = []
    for n in [int(x) for x in a.clients.split(",")]:
        rounds = [run_round(url, a.model, n, a.max_tokens) for _ in range(a.runs)]
        good = [r for r in rounds if "aggregate_tok_s" in r]
        if not good:
            print(f"[{a.label}] clients={n}: 全部失敗 {rounds}"); continue
        med = lambda k: statistics.median(r[k] for r in good)
        row = {"label": a.label, "clients": n, "aggregate_tok_s": med("aggregate_tok_s"),
               "per_req_tok_s": med("per_req_tok_s_mean"), "per_req_min": med("per_req_tok_s_min"),
               "ttft_mean_s": med("ttft_mean_s"), "ttft_max_s": med("ttft_max_s"),
               "errors": sum(len(r["errors"]) for r in rounds)}
        results.append(row)
        print(f"[{a.label}] clients={n:>2}  總吞吐 {row['aggregate_tok_s']:7.1f} tok/s  "
              f"單請求 {row['per_req_tok_s']:6.1f} (最慢 {row['per_req_min']:6.1f})  "
              f"TTFT 平均 {row['ttft_mean_s']:5.2f}s 最久 {row['ttft_max_s']:5.2f}s  錯誤 {row['errors']}", flush=True)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
