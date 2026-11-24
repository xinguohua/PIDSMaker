"""Report how many complete observed types include unresolved or unobserved activities."""
import argparse
from pathlib import Path
import duckdb


def summarize(root):
    db=duckdb.connect(str(root/'raw.duckdb'),read_only=True)
    db.execute('SET threads=4')
    for table,file in [('t','UNIT_TYPES_BY_INSTANCE.csv'),('c','COMPOSITION_COUNTS.csv'),
                       ('parts','COMPONENT_COMPOSITION_FREQUENCIES.csv')]:
        db.execute('CREATE TEMP TABLE '+table+' AS SELECT * FROM read_csv_auto(?)',[str(root/file)])
    db.execute("""CREATE TEMP TABLE typed AS WITH observed AS (SELECT DISTINCT owner FROM events)
      SELECT t.unit_type, count(*) instances,
        count(*) FILTER (WHERE u.subject IS NULL) unknown_actor,
        count(*) FILTER (WHERE u.program='UNKNOWN_PROGRAM') unknown_program,
        count(*) FILTER (WHERE o.owner IS NULL) no_owned_events
      FROM t JOIN units u USING(uid) LEFT JOIN observed o ON o.owner=u.uid GROUP BY t.unit_type""")
    if db.execute("""SELECT count(*) FROM typed WHERE
            unknown_actor NOT IN (0,instances) OR unknown_program NOT IN (0,instances)
            OR no_owned_events NOT IN (0,instances)""").fetchone()[0]:
        raise ValueError('One type mixes incompatible actor, program or event-observation states')
    db.execute("""CREATE TEMP TABLE status AS
      SELECT p."order",p.hop,p.composition_type,
        bool_and(t.unknown_actor=0 AND t.unknown_program=0 AND t.no_owned_events=0) AS all_identified_and_active
      FROM parts p JOIN typed t USING(unit_type) GROUP BY ALL""")
    query="""SELECT c."order",c.hop,all_identified_and_active, count(*) multi_types,
      sum(composition_frequency)::BIGINT multi_instances,
      count(*) FILTER (WHERE composition_frequency=1) types_seen_once,
      count(*) FILTER (WHERE min_component_frequency>composition_frequency) types_min_marginal_gt_joint,
      max(min_component_frequency) FILTER (WHERE composition_frequency=1) maximum_min_marginal_among_once,
      median(min_component_frequency) FILTER (WHERE composition_frequency=1) median_min_marginal_among_once
      FROM c JOIN status USING("order",hop,composition_type) WHERE units>1 GROUP BY ALL ORDER BY 1,2,3"""
    db.sql(query).write_csv(str(root/'OBSERVED_ACTIVITY_DISTRIBUTION.csv'),header=True)
    distinct_query="""SELECT c."order",c.hop,count(*) multi_types,
      sum(composition_frequency)::BIGINT multi_instances,
      count(*) FILTER (WHERE composition_frequency=1) types_seen_once,
      count(*) FILTER (WHERE min_component_frequency>composition_frequency) types_min_marginal_gt_joint,
      count(*) FILTER (WHERE min_component_frequency=1 AND composition_frequency=1) once_with_once_constituent,
      max(min_component_frequency) FILTER (WHERE composition_frequency=1) maximum_min_marginal_among_once,
      median(min_component_frequency) FILTER (WHERE composition_frequency=1) median_min_marginal_among_once
      FROM c JOIN status USING("order",hop,composition_type)
      WHERE all_identified_and_active AND distinct_unit_types>=2 GROUP BY ALL ORDER BY 1,2"""
    db.sql(distinct_query).write_csv(str(root/'IDENTIFIED_DISTINCT_TYPE_DISTRIBUTION.csv'),header=True)
    db.sql('SELECT * FROM typed ORDER BY unit_type').write_csv(str(root/'UNIT_TYPE_OBSERVABILITY.csv'),header=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path)
    summarize(parser.parse_args().root.resolve())
