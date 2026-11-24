"""Tabulate distinct observed activities and collect compact, lossless frequency tables."""
import argparse
from pathlib import Path

import duckdb


def report(root):
    db=duckdb.connect()
    for name,file in [('counts','COMPOSITION_COUNTS'),('components','COMPONENT_COMPOSITION_FREQUENCIES')]:
        db.execute(f'CREATE TEMP TABLE {name} AS SELECT * FROM read_csv_auto(?)',[str(root/(file+'.csv'))])
        dest=root/(file+'.parquet')
        db.sql('SELECT * FROM '+name).write_parquet(str(dest),compression='zstd')
        # Preserve repeated rows, numbers and full signatures across the transfer format.
        for left,right in [(name,'read_parquet(?)'),('read_parquet(?)',name)]:
            if db.execute(f'SELECT count(*) FROM (SELECT * FROM {left} EXCEPT ALL SELECT * FROM {right})',[str(dest)]).fetchone()[0]:
                raise ValueError(f'Lossless conversion failed for {name}')
    db.execute("""CREATE TEMP TABLE activity AS
        SELECT "order",hop,composition_type,
               bool_and(program<>'UNKNOWN_PROGRAM') known_programs
        FROM components GROUP BY ALL""")
    summary=db.sql("""SELECT c."order",c.hop,
         distinct_unit_types>=2 AS different_activity_types,
         known_programs AND unresolved_actor_units=0 AS identified,
         count(*) types,sum(composition_frequency)::BIGINT instances,
         count(*) FILTER(WHERE composition_frequency=1) observed_once,
         count(*) FILTER(WHERE min_component_frequency>composition_frequency) min_above_joint,
         count(*) FILTER(WHERE min_component_frequency=composition_frequency) min_equal_joint,
         count(*) FILTER(WHERE min_component_frequency=1 AND composition_frequency=1) once_with_once_component,
         max(min_component_frequency) FILTER(WHERE composition_frequency=1) largest_minimum_for_once
         FROM counts c JOIN activity USING("order",hop,composition_type)
         WHERE units>1 GROUP BY ALL ORDER BY 1,2,3,4""")
    summary.write_csv(str(root/'DISTINCT_ACTIVITY_SUPPORT.csv'),header=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    report(p.parse_args().root.resolve())
