"""Проверка: послойный прогон обязан повторить замеры полной модели.

Сравнивает строки results/large.csv с results/baseline.csv для одной и той же
модели и конфигурации. Сначала выполните:
    python run_large.py --model tinyllama --bits 16
    python run_large.py --model tinyllama --bits 4 --group-size 128
"""
import csv

TOL = 0.005      # допустимое расхождение ppl (порядок недетерминизма fp16-ядер)


def load(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def key(r):
    bits = int(r["bits"])
    if bits >= 16:
        return (r["model"], 16, 0, "False")
    return (r["model"], bits, int(r["group_size"]), r["sym"])


base = {key(r): float(r["ppl"]) for r in load("results/baseline.csv")}
ok = True
for r in load("results/large.csv"):
    k = key(r)
    if k not in base:
        continue
    d = float(r["ppl"]) - base[k]
    flag = "OK " if abs(d) <= TOL else "!!!"
    ok &= abs(d) <= TOL
    print(f"{flag} {k}: полная модель {base[k]:.4f}  послойно {float(r['ppl']):.4f}  Δ = {d:+.4f}")

print("\nМетодика совпадает" if ok else "\nЕсть расхождения — пришлите вывод")
