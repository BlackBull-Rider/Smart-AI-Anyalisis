with open('backend/providers/yahoo_provider.py', 'r') as f:
    lines = f.readlines()

with open('backend/providers/yahoo_provider.py', 'w') as f:
    for line in lines:
        if "return sorted(rows, key=lambda x: (x[\"fiscal_year\"], x[\"fiscal_quarter\"]), reverse=True)" in line:
            f.write('        for r in rows:\n')
            f.write('            r.pop("period_type", None)\n')
        f.write(line)

print("✅ 'period_type' bug fixed!")
