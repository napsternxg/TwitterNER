import requests
import json
import regex as re

query="""
SELECT ?musician ?musicianLabel 
WHERE {
  ?musician wdt:P31 wd:Q5 . #instance of human
        ?musician wdt:P106/wdt:P279 wd:Q639669 . #occupation a subclass of musician
  SERVICE wikibase:label {
    bd:serviceParam wikibase:language "en" .
   }
}

LIMIT 10
"""

QID_REGEX=re.compile(r'^Q[0-9]+$')


url = 'https://query.wikidata.org/bigdata/namespace/wdq/sparql'
def download_data(queryfile,outdir="./downloaded"):
    query = ""
    with open(queryfile) as fp:
        query = fp.read().strip()
    print query
    data = requests.get(url, params={'query': query, 'format': 'json'}).json()
    print "Downloaded %s records for %s" % (len(data["results"]["bindings"]), queryfile)
    outfile="%s/%s.json" % (outdir,queryfile)
    print "Saving in %s" % outfile
    fp.close()
    with open(outfile, "wb+") as fp:
        for k in data["results"]["bindings"]:
            try:
                line=k["musicLabel"]["value"]
                if QID_REGEX.match(line):
                    continue
                print >> fp, line
            except:
                continue
    return data



