# coding: utf-8
import re, sys
src = 'src/otf_backtest_engine.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()
print('Original length:', len(content))
print('Has execution_calendar param:', 'execution_calendar' in content)