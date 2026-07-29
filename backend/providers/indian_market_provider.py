import logging
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import date, timedelta

try:
    from jugaad_data.nse import NSELive, stock_df
    JUGAAD_AVAILABLE = True
except ImportError as e:
    print(f"Import Error: {e}")
    JUGAAD_AVAILABLE = False

class IndianMarketProvider:
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = self._create_robust_session()
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        }

    def _create_robust_session(self):
        session = requests.Session()
        retry = Retry(connect=3, read=3, backoff_factor=0.5, status_forcelist=[429, 500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('http://', adapter)
        session.mount('https://', adapter)
        return session

    # ==========================================
    # 1. NSE DATA (Live + Smart CSV Fallback)
    # ==========================================
    def fetch_nse_data(self, symbol: str) -> dict:
        result = {
            "delivery_percent": np.nan,
            "delivery_quantity": np.nan
        }
        clean_symbol = symbol.upper().replace('.NS', '')
        
        if not JUGAAD_AVAILABLE:
            return result
            
        # First Try: Live API
        try:
            n = NSELive()
            data = n.stock_quote(clean_symbol)
            if 'securityWiseDP' in data:
                dp = data['securityWiseDP']
                result['delivery_quantity'] = float(dp.get('deliveryQuantity', np.nan))
                result['delivery_percent'] = float(dp.get('deliveryToTradedQuantity', np.nan))
                return result
        except Exception as e:
            self.logger.warning(f"Live API Blocked. Switching to EOD CSV Fallback...")
        
        # Second Try: Historical CSV API
        try:
            end_dt = date.today()
            start_dt = end_dt - timedelta(days=7) # Look back 7 days
            
            df = stock_df(symbol=clean_symbol, from_date=start_dt, to_date=end_dt, series="EQ")
            if not df.empty:
                latest_row = df.iloc[0]
                
                # আপডেট করা কলাম লিস্ট (তোর প্রিন্ট করা আউটপুট অনুযায়ী)
                deliv_qty_keys = ['DELIVERY QTY', 'Deliverable Qty', 'Deliverable Volume', 'Delivery Qty', 'DELIV_QTY']
                deliv_pct_keys = ['DELIVERY %', '% Dly Qt to Traded Qty', '% Deli. Qty to Traded Qty', 'Delivery To Traded Qty', 'DELIV_PER']
                
                dq_key = next((k for k in deliv_qty_keys if k in df.columns), None)
                dp_key = next((k for k in deliv_pct_keys if k in df.columns), None)
                
                if dq_key and dp_key:
                    # Remove commas and convert
                    dq_val = str(latest_row[dq_key]).replace(',', '').strip()
                    dp_val = str(latest_row[dp_key]).replace(',', '').replace('%', '').strip()
                    
                    result['delivery_quantity'] = float(dq_val) if dq_val and dq_val != 'nan' else np.nan
                    result['delivery_percent'] = float(dp_val) if dp_val and dp_val != 'nan' else np.nan
                    self.logger.info("✅ Successfully fetched NSE delivery from CSV Fallback!")
                else:
                    self.logger.warning("❌ Delivery columns not found even after update!")
        except Exception as e:
            self.logger.warning(f"CSV Fallback also failed: {str(e)[:50]}")
            
        return result

    # ==========================================
    # 2. SCREENER.IN DATA (100% Working)
    # ==========================================
    def fetch_screener_data(self, symbol: str) -> dict:
        result = {
            "roce": np.nan,
            "promoter_holding": np.nan,
            "fii_holding": np.nan,
            "dii_holding": np.nan,
            "public_holding": np.nan
        }
        
        clean_symbol = symbol.upper().replace('.NS', '')
        url = f"https://www.screener.in/company/{clean_symbol}/consolidated/"
        
        try:
            response = self.session.get(url, headers=self.headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                ratios = soup.find_all('li', class_='flex flex-space-between')
                for ratio in ratios:
                    name_tag = ratio.find('span', class_='name')
                    if not name_tag: continue
                    name = name_tag.text.strip().lower()
                    value_span = ratio.find('span', class_='number')
                    if not value_span: continue
                    value = value_span.text.replace(',', '').strip()
                    if 'roce' in name:
                        result['roce'] = float(value) if value else np.nan

                shareholding_section = soup.find('section', id='shareholding')
                if shareholding_section:
                    table = shareholding_section.find('table')
                    if table:
                        rows = table.find_all('tr')
                        for row in rows:
                            cols = row.find_all('td')
                            if len(cols) > 1:
                                name = cols[0].text.strip().lower()
                                val = cols[-1].text.strip()
                                try:
                                    val_float = float(val.replace('%', ''))
                                    if 'fii' in name: result['fii_holding'] = val_float
                                    elif 'dii' in name: result['dii_holding'] = val_float
                                    elif 'promoters' in name: result['promoter_holding'] = val_float
                                    elif 'public' in name: result['public_holding'] = val_float
                                except:
                                    pass
        except Exception:
            pass
            
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    provider = IndianMarketProvider()
    
    print("\nFetching NSE Data (Live + CSV Fallback)...")
    nse_data = provider.fetch_nse_data("RELIANCE")
    print(nse_data)
