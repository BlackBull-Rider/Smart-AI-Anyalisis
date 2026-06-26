import pandas as pd
from backend.ai.master_engine import generate_final_master_report

def run_reliance_test():
    print("--- TESTING SYSTEM: RELIANCE INDUSTRIES ---")
    
    # Reliance এর জন্য ডামি ডেটা (তুমি এখানে তোমার CSV ফাইল লোড করতে পারো)
    # df = pd.read_csv("reliance_data.csv") 
    
    # স্যাম্পল ডেটা স্ট্রাকচার (Testing logic check)
    data = {
        'open': [2500, 2510, 2505, 2520, 2530],
        'high': [2520, 2525, 2515, 2540, 2550],
        'low': [2490, 2500, 2495, 2510, 2520],
        'close': [2510, 2505, 2515, 2535, 2545],
        'volume': [100000, 150000, 120000, 200000, 250000]
    }
    df = pd.DataFrame(data)
    
    account_balance = 500000 # 5 Lakh Capital
    
    print("Processing Reliance Data through Master Engine...")
    report = generate_final_master_report(df, account_balance)
    
    # রিপোর্ট প্রিন্ট করো
    print("\n" + "="*50)
    print("RELIANCE AI REPORT")
    print("="*50)
    for category, details in report.items():
        print(f"\n[ {category} ]")
        print(details)
    print("="*50)

if __name__ == "__main__":
    run_reliance_test()
