import duckdb
from quantswing.core.config import get_config
from quantswing.store.repositories import DataPaths

p = DataPaths(get_config())
con = duckdb.connect(str(p.duckdb))
df = con.execute("""
    SELECT symbol, COUNT(*) dup_rows
    FROM prices_daily
    WHERE (symbol, date) IN (
        SELECT symbol, date FROM prices_daily GROUP BY symbol, date HAVING COUNT(*) > 1
    )
    GROUP BY symbol ORDER BY dup_rows DESC
""").fetchdf()
print(len(df), "symbols affected")
print(df.to_string())
