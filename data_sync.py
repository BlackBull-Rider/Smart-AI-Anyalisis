"""
GREEN BULL RIDER V6
Script: data_sync.py
Master Data Ingestion Pipeline
"""

import logging
from backend.providers.yahoo_provider import YahooProvider
from backend.repository.stock_repository import repository
from backend.universe.universe_loader import UniverseLoader
from backend.config.settings import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("DataSync")

def sync_universe_to_master():
    logger.info("🔄 Running UniverseLoader to fetch active symbols from NSE...")
    try:
        loader = UniverseLoader(str(settings.database_path))
        count = loader.refresh()
        logger.info(f"✅ Successfully synced {count} active symbols to Stock Master.")
    except Exception as e:
        logger.error(f"❌ Failed to sync universe: {e}")

def run_sync():
    provider = YahooProvider()
    active_stocks = repository.get_active_symbols()
    total = len(active_stocks)
    
    if total == 0:
        logger.error("❌ Stock Master is empty!")
        return
    
    logger.info(f"🚀 Starting Fundamental Sync for {total} stocks...")
    
    for idx, stock in enumerate(active_stocks, 1):
        symbol = stock["symbol"]
        logger.info(f"[{idx}/{total}] ⚡ Processing: {symbol}")
        
        try:
            # 1. Profile
            profile = provider.get_company_info(symbol)
            if profile and isinstance(profile, dict):
                profile["symbol"] = symbol
                repository.save_company_profile(profile)

            # 2. Fundamentals
            fund = provider.get_fundamentals(symbol)
            if fund and isinstance(fund, dict):
                fund["symbol"] = symbol
                repository.save_fundamental(fund)

            # 3. Financials
            fins = provider.get_financials(symbol)
            if fins and isinstance(fins, list) and len(fins) > 0:
                for f in fins: f["symbol"] = symbol
                repository.save_financials(fins)

            # 4. Shareholders
            sh = provider.get_share_holders(symbol)
            if sh and isinstance(sh, list) and len(sh) > 0:
                for s in sh: s["symbol"] = symbol
                repository.save_shareholding(sh)

            # 5. Actions (✅ NOT NULL FIX APPLIED HERE)
            acts = provider.get_actions(symbol)
            if acts and isinstance(acts, list) and len(acts) > 0:
                for a in acts: a["symbol"] = symbol
                repository.save_corporate_actions(acts)

            # 6. Analyst
            analyst = provider.get_analyst_targets(symbol)
            if analyst and isinstance(analyst, dict):
                analyst["symbol"] = symbol
                repository.save_analyst_data(analyst)

            logger.info(f"  └── ✅ {symbol} synced.")
            
        except Exception as e:
            logger.error(f"  └── ❌ Failed {symbol}: {e}")

if __name__ == "__main__":
    sync_universe_to_master()
    run_sync()
