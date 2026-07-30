import logging
from backend.pipeline.market_pipeline import MarketPipeline

# লগিং সেটআপ
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

def show_progress(current, total, info):
    print(f"🔄 [{current}/{total}] {info}")

print(f"\n{'='*80}")
print(" 🚀 FIRING LAYER-1 PIPELINE FOR ALL ACTIVE SYMBOLS IN UNIVERSE")
print(f"{'='*80}\n")

with MarketPipeline() as pipeline:
    # ইউনিভার্সের সব একটিভ স্টকের ওপর FULL SYNC চলবে
    report = pipeline.full_sync(progress_callback=show_progress)

print(f"\n{'='*80}")
print(f" ✅ FULL UNIVERSE LAYER-1 EXECUTION COMPLETE")
print(f"{'='*80}")
print(f"  Total Symbols Requested : {report.total_symbols}")
print(f"  Successfully Synced     : {report.successful}")
print(f"  Failed Symbols Count    : {report.failed}")
print(f"  Total Time Elapsed      : {report.total_elapsed_sec} sec")

if report.failed_symbols:
    print(f"\n🚨 Failed Symbols List  : {report.failed_symbols}")
print(f"{'='*80}\n")
