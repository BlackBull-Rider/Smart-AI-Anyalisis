with open('backend/providers/yahoo_provider.py', 'r') as f:
    content = f.read()

# Change history fetch limit from 1 year to 5 years (1825 days)
content = content.replace('timedelta(days=365)', 'timedelta(days=1825)')

with open('backend/providers/yahoo_provider.py', 'w') as f:
    f.write(content)

print("✅ Yahoo Provider History Limit increased to 5 Years!")
