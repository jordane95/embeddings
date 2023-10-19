import json
from collections import defaultdict

lang = 'python'

benchmark_data = json.load(open(f'csn-{lang}/benchmarks.json'))


matched_results = json.load(open(f'csn-{lang}/excluded-data.json'))

# print(matched_results)


# "filter_reason": "sods_match", "matched_substring": "ValueError: invalid literal for float():"

contaminated = defaultdict(set)

for result in matched_results:
    contaminated[result['filter_reason'].replace('_match', '')].add(result['matched_substring'])


print("Benchmark data:")
for benchmark, values in benchmark_data.items():
    print(f"num strings from {benchmark}: {len(contaminated[benchmark]) / len(values) * 100:.2f}%")

