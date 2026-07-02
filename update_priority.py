import re

with open('backend/analyzers/pattern_analyzer.py', 'r') as file:
    data = file.read()

new_priority = '"priority": {"TRIANGLE": 1, "WEDGE": 2, "FLAG": 3, "PENNANT": 4, "CHANNEL": 5, "RECTANGLE": 6, "HEAD_SHOULDERS": 7, "INV_HEAD_SHOULDERS": 8, "CUP_HANDLE": 9, "DOUBLE_TOP": 10, "DOUBLE_BOTTOM": 10, "TRIPLE_TOP": 11, "TRIPLE_BOTTOM": 11, "ROUNDING_TOP": 12, "ROUNDING_BOTTOM": 12}'

data = re.sub(r'"priority":\s*\{.*?\}', new_priority, data, flags=re.DOTALL)

with open('backend/analyzers/pattern_analyzer.py', 'w') as file:
    file.write(data)
print("Priority updated successfully!")
