from backend.registry.dynamic_registry import registry

def test():
    # কোনো র‍্যান্ডম ইন্ডিকেটর টেস্ট কর
    rsi_func = registry.get("momentum.rsi")
    if rsi_func:
        print("✅ Momentum RSI found!")
    else:
        print("❌ RSI missing!")

    # তোর সব কি (keys) প্রিন্ট করে দেখ
    print(f"\nTotal loaded: {len(registry._registry)}")
    # প্রথম ১০টা প্রিন্ট করে দেখ
    print("Sample keys:", list(registry._registry.keys())[:10])

if __name__ == "__main__":
    test()
