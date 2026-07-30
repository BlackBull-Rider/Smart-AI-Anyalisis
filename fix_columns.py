with open('backend/providers/yahoo_provider.py', 'r') as f:
    content = f.read()

if 'r.pop("report_date", None)' not in content:
    content = content.replace(
        'r.pop("period_type", None)',
        'r.pop("period_type", None)\n            r.pop("report_date", None)\n            r.pop("working_capital", None)'
    )

with open('backend/providers/yahoo_provider.py', 'w') as f:
    f.write(content)

print("✅ 'report_date' and 'working_capital' removed before DB insert!")
