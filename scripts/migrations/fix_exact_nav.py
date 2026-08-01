src = 'tests/test_holiday_share_adjustment.py'
with open(src, 'r', encoding='utf-8') as f:
    content = f.read()

# Use exact values to avoid reconciliation error
old_nav_rows = """    nav_rows = [
        # Date T: normal, NAV=1.0
        ("H001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        # Holiday H: NAV drops to 0.952381 but adj=1.05 means total return unchanged
        # reconstructed_return = 0.952381/1.0 * 1.05 - 1 = 0.0
        ("H001", "2024-06-29", 0.952381, 1.0, 0.0, 0.0, 1.05, 1.0),
        # Date T+1: NAV grows to 0.98 from 0.952381, adj=1.0
        # reconstructed_return = 0.98/0.952381 * 1.0 - 1 = 0.028797
        ("H001", "2024-06-30", 0.98, 1.0, 2.8797, 0.0, 1.0, 1.0),
    ]"""

new_nav_rows = """    # Use exact values for reconciliation: NAV(H) = NAV(T)/adj(H) so total return unchanged
    nav_h = 1.0 / 1.05
    daily_growth_t1 = (0.98 / nav_h * 1.0 - 1.0) * 100.0
    nav_rows = [
        # Date T: normal, NAV=1.0
        ("H001", "2024-06-28", 1.0, 1.0, 0.0, 0.0, 1.0, 1.0),
        # Holiday H: NAV drops but adj=1.05 means total return unchanged
        ("H001", "2024-06-29", nav_h, 1.0, 0.0, 0.0, 1.05, 1.0),
        # Date T+1: NAV grows from holiday NAV, adj=1.0
        ("H001", "2024-06-30", 0.98, 1.0, daily_growth_t1, 0.0, 1.0, 1.0),
    ]"""

content = content.replace(old_nav_rows, new_nav_rows)

with open(src, 'w', encoding='utf-8') as f:
    f.write(content)

print("Fixed with exact NAV values")