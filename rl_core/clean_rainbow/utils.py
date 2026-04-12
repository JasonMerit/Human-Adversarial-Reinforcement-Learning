import time, json
from collections import defaultdict
from functools import wraps
from rich import print

class TimerRegistry:
    _total = defaultdict(float)
    _calls = defaultdict(int)
    _t0 = time.time()
    _time = 0
    _active = False

    @staticmethod
    def record(name, dt):
        TimerRegistry._total[name] += dt
        TimerRegistry._calls[name] += 1
    
    @staticmethod
    def start():
        if TimerRegistry._active:
            raise RuntimeError("Timer already active")
        TimerRegistry._time = time.perf_counter()
        TimerRegistry._active = True
    
    @staticmethod
    def stop(name):
        if not TimerRegistry._active:
            raise RuntimeError(f"Timer not active for {name}")
        TimerRegistry.record(name, time.perf_counter() - TimerRegistry._time)
        TimerRegistry._active = False

    @staticmethod
    def wrap_fn(name):
        def decorator(fn):

            @wraps(fn)
            def wrapped(*args, **kwargs):
                start = time.perf_counter()
                out = fn(*args, **kwargs)
                TimerRegistry.record(name, time.perf_counter() - start)
                return out

            return wrapped

        return decorator
    
    @staticmethod
    def report():
        print(f"\nTiming Report over {int(time.time() - TimerRegistry._t0)}s:")

        print(f"{'name':20s} | {'calls':>8s} | {'total(s)':>10s} | {'avg(ms)':>10s}")
        print("-" * 60)

        # Sort by average
        items = sorted(TimerRegistry._total.items(), key=lambda x: x[1]/TimerRegistry._calls[x[0]], reverse=True)

        for name, total in items:
            calls = TimerRegistry._calls[name]
            print(f"{name:20s} | {calls:8d} | {total:10.3f} | {total / calls*1000:10.3f}")
    
    @staticmethod
    def export(path):
        with open(path, "w") as f:
            json.dump({
                name: {
                    "calls": TimerRegistry._calls[name],
                    "total": TimerRegistry._total[name],
                    "avg": TimerRegistry._total[name] / TimerRegistry._calls[name],
                }
                for name in TimerRegistry._total
            }, f, indent=4)