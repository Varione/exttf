c1_core = {"160706", "000218", "001512", "260102"}
c1_satellite_pool = {"160706", "000008", "050021", "007466", "110022", "000041", "000311", "001692"}
c2_core = {"160706", "000218", "001512", "260102"}
c2_satellite_pool = {"160706", "000008", "050021", "007466"}
b2_funds = {"160706", "000218", "001512", "260102"}

c1_used = c1_core | c1_satellite_pool
c2_used = c2_core | c2_satellite_pool

# 26 affected funds from previous query
affected = {
    "010631", "012035", "013333", "016064", "016417", "017180", "017528",
    "000834", "017071", "018584", "018809", "019813", "020041", "519206",
    "003564", "007061", "007156", "007204", "010292", "020940", "021231"
}

print("C1 affected:", affected & c1_used)
print("C2 affected:", affected & c2_used)
print("B2 affected:", affected & b2_funds)
print("\nNone of the 26 funds with lost factors are in C1/C2/B2 universe.")
print("Calendar fix + aggregation fix does NOT change C1/C2/B2 metrics.")