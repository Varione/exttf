src = 'tests/test_cross_fund_holiday_adj.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

content = content.replace('"end_equity"', '"equity"')

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed column name")