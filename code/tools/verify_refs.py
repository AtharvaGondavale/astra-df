"""Check every references.bib entry against Crossref (title, year, venue, DOI).

Prints one line per entry with the best Crossref match and a similarity score;
entries scoring below 0.9 or differing in year need a manual look.
    python tools/verify_refs.py ../paper/references.bib > ../results/ref_check.tsv
"""
import difflib, json, re, sys, time, urllib.error, urllib.parse, urllib.request

def parse(path):
    txt = open(path).read()
    for m in re.finditer(r"@(\w+)\s*{\s*([^,]+),(.*?)\n}", txt, re.S):
        typ, key, body = m.groups()
        f = {k.lower(): re.sub(r"[{}\\]|\s+", " ", v).strip()
             for k, v in re.findall(r"(\w+)\s*=\s*{((?:[^{}]|{[^{}]*})*)}", body)}
        yield key.strip(), typ, f

def norm(s):
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("'", "")).strip()

def query(title, author):
    q = urllib.parse.urlencode({"query.bibliographic": f"{title} {author}", "rows": 3,
                                "select": "title,author,issued,container-title,DOI,type"})
    req = urllib.request.Request(f"https://api.crossref.org/works?{q}",
                                 headers={"User-Agent": "ref-check (mailto:research@example.org)"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.load(r)["message"]["items"]
        except urllib.error.HTTPError as e:
            if e.code not in (429, 500, 502, 503) or attempt == 5:
                raise
            time.sleep(5 * (attempt + 1))

print("key\tsim\tbib_year\tcr_year\tbib_doi\tcr_doi\tcr_venue\tcr_title")
for key, typ, f in parse(sys.argv[1]):
    title = f.get("title", "")
    if not title or key == "c2pa2023":
        print(f"{key}\t-\t{f.get('year','')}\t-\t-\t-\t-\t(skipped: non-paper)"); continue
    author = f.get("author", "").split(" and ")[0].split(",")[0]
    try:
        items = query(title, author)
    except Exception as e:
        print(f"{key}\tERR\t{e}"); continue
    best, bs = None, 0
    for it in items:
        t = (it.get("title") or [""])[0]
        s = difflib.SequenceMatcher(None, norm(title), norm(t)).ratio()
        if s > bs: best, bs = it, s
    if best is None:
        print(f"{key}\t0\t{f.get('year','')}"); continue
    y = (best.get("issued", {}).get("date-parts") or [[None]])[0][0]
    print(f"{key}\t{bs:.2f}\t{f.get('year','')}\t{y}\t{f.get('doi','')}\t{best.get('DOI','')}\t"
          f"{(best.get('container-title') or [''])[0][:60]}\t{(best.get('title') or [''])[0][:90]}")
    sys.stdout.flush(); time.sleep(2.0)
