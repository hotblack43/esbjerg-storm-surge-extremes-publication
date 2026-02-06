import pyreadr, sys
df = next(iter(pyreadr.read_r(sys.argv[1]).values()))
df.to_csv(sys.argv[2], index=False)
# usage: python rds2csv.py input.rds output.csv

