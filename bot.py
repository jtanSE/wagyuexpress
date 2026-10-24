import argparse
import getpass
import http.client
import json
import math
import os
import sys
import time
from pathlib import Path

HOST = "nightshift.joinbuilders.org"
SPOTS = ("oval", "thompson", "union", "stadium", "highst", "northres")
COSTS = {"cheap": 2.5, "standard": 3.75, "premium": 5.5}
RANK, START, STEP, SLOW, FLOOR, CAP = 3, 10.0, .25, .6, 8.75, 15.0


def number(value, default=0.0):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def rules(state):
    r = state.get("rules") if isinstance(state.get("rules"), dict) else {}
    spots = r.get("spots") if isinstance(r.get("spots"), list) and r.get("spots") else list(SPOTS)
    costs = r.get("costs") if isinstance(r.get("costs"), dict) else COSTS
    return {"spots": spots, "budget": number(r.get("budget"), 500.0),
            "premium": number(costs.get("premium"), 5.5) or 5.5,
            "low": number(r.get("price_min"), 1.0), "high": number(r.get("price_max"), 30.0)}


def valid(order, state):
    r = rules(state)
    try:
        return (order["spot"] in r["spots"] and order["quality"] == "premium"
                and r["low"] <= float(order["price"]) <= r["high"]
                and isinstance(order["meals"], int) and 0 <= order["meals"]
                and order["meals"] * r["premium"] <= r["budget"] + 1e-9)
    except (KeyError, TypeError, ValueError):
        return False


class Camp:
    def __init__(self):
        self.home = {}
        self.price = {}

    def decide(self, state):
        r = rules(state)
        key = (state.get("mode"), state.get("market"))
        last = state.get("last_day") if isinstance(state.get("last_day"), dict) else {}
        mine = last.get("you") if isinstance(last.get("you"), dict) else {}
        you = state.get("you") if isinstance(state.get("you"), dict) else {}
        fresh = int(number(state.get("day"))) == 0 or not last.get("day")
        home = self.home.get(key)
        if home not in r["spots"]:
            seen = you.get("last_spot") or mine.get("spot")
            if not fresh and seen in r["spots"]:
                home = seen
            else:
                forecast = state.get("forecast") if isinstance(state.get("forecast"), dict) else {}
                expected = forecast.get("expected_customers") if isinstance(forecast.get("expected_customers"), dict) else {}
                ranked = sorted(r["spots"], key=lambda s: (-number(expected.get(s)), r["spots"].index(s)))
                home = ranked[min(RANK, len(ranked)) - 1]
            self.home[key] = home
        price = self.price.get(key, START)
        if not fresh and mine.get("spot") == home and number(mine.get("price")) > 0 and number(mine.get("meals")) > 0:
            price = number(mine["price"])
            if number(mine.get("sold")) >= number(mine["meals"]):
                price += STEP
            elif number(mine.get("sold")) < SLOW * number(mine["meals"]):
                price -= STEP
        price = round(max(FLOOR, r["low"], min(CAP, r["high"], price)), 2)
        self.price[key] = price
        meals = int((r["budget"] + 1e-9) / r["premium"])
        staff = you.get("staff") if isinstance(you.get("staff"), dict) else {}
        capacity = number(staff.get("capacity"))
        if capacity > 0:
            meals = min(meals, int(capacity))
        return {"spot": home, "quality": "premium", "price": price, "meals": meals}

    def fallback(self, state):
        r = rules(state)
        home = self.home.get((state.get("mode"), state.get("market")))
        return {"spot": home if home in r["spots"] else r["spots"][0], "quality": "premium",
                "price": max(r["low"], min(r["high"], START)), "meals": int((r["budget"] + 1e-9) / r["premium"])}


class Transport:
    def __init__(self, key):
        self.key = key
        self.connection = None
        self.headers = {}

    def call(self, method, path, body=None, timeout=32):
        if self.connection is None:
            self.connection = http.client.HTTPSConnection(HOST, timeout=timeout)
        self.connection.timeout = timeout
        if self.connection.sock:
            self.connection.sock.settimeout(timeout)
        try:
            self.connection.request(method, path, body=json.dumps(body).encode() if body is not None else None,
                                    headers={"Authorization": "Bearer " + self.key, "Content-Type": "application/json"})
            response = self.connection.getresponse()
            self.headers = {k.lower(): v for k, v in response.getheaders()}
            payload = response.read()
            data = json.loads(payload) if payload else {}
            return response.status, data if isinstance(data, dict) else {}
        except Exception:
            self.close()
            raise

    def close(self):
        if self.connection:
            self.connection.close()
        self.connection = None


def play(transport, policy, base):
    version, sent = None, None
    while True:
        try:
            status, state = transport.call("GET", base + "/state" + ("?since=%s" % version if version is not None else ""))
        except Exception as e:
            print("Connection error (%s), retrying" % type(e).__name__, file=sys.stderr)
            time.sleep(1)
            continue
        if status == 429:
            time.sleep(max(.2, min(30, number(transport.headers.get("retry-after"), 1.0))))
            continue
        if status in (401, 404):
            raise RuntimeError("State access rejected (%d)" % status)
        if status != 200 or "phase" not in state:
            time.sleep(.5)
            continue
        version = state.get("version")
        if state.get("phase") == "over" and state.get("mode") == "practice":
            print(json.dumps({"event": "practice_over", "cash": (state.get("you") or {}).get("cash")}))
            return state
        key = (state.get("mode"), state.get("market"), state.get("order_for_day"))
        if state.get("phase") != "night" or not state.get("orders_open") or key == sent:
            continue
        try:
            order = policy.decide(state)
        except Exception as e:
            print("Strategy error (%s), using fallback" % type(e).__name__, file=sys.stderr)
            order = policy.fallback(state)
        if not valid(order, state):
            order = policy.fallback(state)
        if (state.get("you") or {}).get("order") == order:
            sent = key
            continue
        try:
            status, result = transport.call("POST", base + "/orders", order, timeout=5)
        except Exception as e:
            print("Order error (%s), retrying" % type(e).__name__, file=sys.stderr)
            version = None
            continue
        if status in (200, 201):
            sent = key
            print(json.dumps({"event": "order_accepted", "round": list(key), "order": order}))
        elif status == 429:
            time.sleep(max(.2, min(30, number(transport.headers.get("retry-after"), .25))))
            version = None
        elif 400 <= status < 500:
            sent = key
            print(json.dumps({"event": "order_rejected", "round": list(key), "status": status, "result": result}))
        else:
            version = None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Wagyu Express")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--state", type=Path)
    mode.add_argument("--live", action="store_true")
    mode.add_argument("--practice", action="store_true")
    ap.add_argument("--allow-orders", action="store_true")
    ap.add_argument("--seed", default=None)
    ap.add_argument("--days", type=int, default=30)
    ap.add_argument("--rivals", type=int, default=6)
    ap.add_argument("--inventory-mode", default=None)
    ap.add_argument("--price-discount", type=float, default=0.0)
    ap.add_argument("--memory", default=None)
    ap.add_argument("--log", default=None)
    args = ap.parse_args(argv)
    policy = Camp()
    if args.state:
        print(json.dumps(policy.decide(json.loads(args.state.read_text())), indent=2))
        return 0
    if not (args.live or args.practice):
        ap.print_help()
        return 0
    if not args.allow_orders:
        ap.error("Connecting to play requires --allow-orders; nothing was sent")
    key = os.environ.get("NIGHTSHIFT_KEY", "").strip()
    key_file = Path(".nightshift/truck-key")
    if not key and key_file.exists():
        key = key_file.read_text().strip()
    if not key and sys.stdin.isatty():
        key = getpass.getpass("Existing team truck key (hidden): ").strip()
    if not key:
        ap.error("NIGHTSHIFT_KEY is missing; nothing was sent")
    transport = Transport(key)
    base = "/api/v1"
    try:
        if args.practice:
            body = {"days": args.days, "rivals": args.rivals}
            if args.seed is not None:
                body["seed"] = args.seed
            status, result = transport.call("POST", "/api/v1/practice", body)
            if status not in (200, 201) or not str(result.get("base", "")).startswith("/api/v1/practice/"):
                raise RuntimeError("Could not create a practice game: HTTP %s" % status)
            base = result["base"].rstrip("/")
        play(transport, policy, base)
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
