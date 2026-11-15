#!/bin/bash
# catalog-size sweep: same examples and training, catalog resized (bench/zpred.py Catalog size=)
cd "$(dirname "$0")/../../.."
for n in 20000 100000 300000 1003622 3000000; do
  echo "== size $n"
  timeout 3000 nice -n 10 .venv/bin/python -I bench/zpred.py run --size $n --settings catalog --workers 8 || echo "size $n failed: $?"
done
