from backend.repository.stock_repository import repository

def show_full_details(symbol: str):
    print(f"\n{'='*70}\n 🕵️ FULL DATABASE RAW DUMP FOR: {symbol}\n{'='*70}")

    # 1. Company Profile
    profile = repository.get_company_profile(symbol)
    print(f"\n🏢 [1] COMPANY PROFILE (All Columns):")
    if profile:
        for k, v in profile.items():
            print(f"   {k}: {v}")
    else:
        print("   ❌ Missing")

    # 2. Fundamentals
    fund = repository.get_fundamental(symbol)
    print(f"\n📈 [2] FUNDAMENTALS (All Columns - Dynamic):")
    if fund:
        for k, v in fund.items():
            if v is not None:  # শুধু যেগুলোতে ডেটা আছে সেগুলো দেখাবে
                print(f"   {k}: {v}")
    else:
        print("   ❌ Missing")

    # 3. Financials (Latest Quarter/Year)
    fins = repository.get_financials(symbol)
    print(f"\n💰 [3] FINANCIALS (Latest Record - All Columns):")
    if fins:
        print(f"   Total Periods Saved: {len(fins)}")
        print(f"   --- Showing Latest Period ---")
        for k, v in fins[0].items():
            if v is not None:
                print(f"   {k}: {v}")
    else:
        print("   ❌ Missing")

    # 4. Shareholding
    share = repository.get_shareholding(symbol)
    print(f"\n🤝 [4] SHAREHOLDING (Latest Record):")
    if share:
        print(f"   Total Quarters Saved: {len(share)}")
        for k, v in share[0].items():
            if v is not None:
                print(f"   {k}: {v}")
    else:
        print("   ❌ Missing")

    # 5. Corporate Actions
    acts = repository.get_corporate_actions(symbol)
    print(f"\n🎁 [5] CORPORATE ACTIONS (All Records):")
    if acts:
        print(f"   Total Actions Saved: {len(acts)}")
        for act in acts:
            print(f"   -> {act['action_date']} | Type: {act['action_type']} | Div: {act.get('dividend')} | Split: {act.get('split_ratio')}")
    else:
        print("   ❌ Missing")

    print(f"\n{'='*70}\n")

if __name__ == '__main__':
    # তুই চাইলে এখানে AAKASH এর বদলে অন্য যেকোনো স্টকের নাম দিতে পারিস
    show_full_details("AAKASH")
