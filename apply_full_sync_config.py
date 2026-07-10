import re

filepath = "backend/pipeline/market_pipeline.py"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # মেইন ব্লককে আপনার দেওয়া নতুন কনফিগারেশন দিয়ে আপডেট করা
    main_block = """
if __name__ == "__main__":
    import datetime
    pipeline = PipelineFactory.create_production_pipeline()

    config = PipelineConfig(
        start_date="2000-01-01",
        end_date=datetime.date.today().isoformat(),
        timeframe="1D",
        exchange="NSE",
        symbols=None,  # 'None' মানেই হলো ডাটাবেস থেকে সব অ্যাক্টিভ স্টক লোড হবে
        batch_size=50
    )

    print(f"🚀 Starting Full Sync for all symbols (Since 2000)...")
    result = pipeline.run(config)
    
    print("=" * 60)
    print("FULL SYNC COMPLETED")
    print("=" * 60)
    print(f"Success : {result.success}")
    if result.summary:
        print(f"Processed : {result.summary.total_processed}")
"""

    # আগের মেইন ব্লকটি খুঁজে নতুনটি দিয়ে রিপ্লেস করা
    pattern = r"if __name__ == \"__main__\":.*"
    new_content = re.sub(pattern, main_block, content, flags=re.DOTALL)

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(new_content)
        
    print("✅ Configuration applied successfully!")
except Exception as e:
    print(f"Error: {e}")
