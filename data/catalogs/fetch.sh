#!/bin/bash
# shallow, sparse clones of public tool/API catalogs (data only; nothing here is executed)
set -u
cd "$(dirname "$0")"
sparse() { # repo dir paths...
  local repo=$1 dir=$2; shift 2
  [ -d $dir/.git ] && { echo "$dir exists"; return; }
  git clone -q --depth 1 --filter=blob:none --sparse https://github.com/$repo.git $dir && git -C $dir sparse-checkout set "$@" && echo "$dir ok $(du -sh $dir | cut -f1)"
}
sparse APIs-guru/openapi-directory apisguru APIs &
sparse boto/botocore botocore botocore/data &
sparse Azure/azure-rest-api-specs azure specification &
sparse microsoftgraph/msgraph-metadata msgraph openapi/v1.0 &
wait
mkdir -p google && cd google && curl -s https://www.googleapis.com/discovery/v1/apis > directory.json && \
python3 -I -c "import json;[print(i['discoveryRestUrl']) for i in json.load(open('directory.json'))['items']]" > urls.txt && \
i=0; while read u; do i=$((i+1)); curl -s --max-time 30 "$u" -o doc_$i.json; done < urls.txt; echo "google ok $(ls doc_*.json | wc -l) docs"
