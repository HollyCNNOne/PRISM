import json
man = json.load(open("/root/autodl-tmp/PRISM/dataset_manifest.json"))
keep = [
    "patients_head1/patient_06",   # complete 183
    "patients_head1/patient_02",   # large 185
    "patients_head1/patient_01",   # medium 90
    "patients_head1/patient_03",   # small 25
    "patients_head2/patient_05",   # complete 178
    "patients_head2/patient_04",   # large 217
    "patients_head2/patient_02",   # medium 106
]
cases = [c for c in man["cases"] if c["path"] in keep]
out = {"root": man["root"], "note": "V2.2 reduced (7 cases)", "cases": cases}
json.dump(out, open("/root/autodl-tmp/PRISM/dataset_manifest_v22.json", "w"), indent=2)
print("kept", len(cases), "cases:", [c["path"] + ":" + c["category"] for c in cases])
