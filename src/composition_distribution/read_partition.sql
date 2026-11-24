-- Run from the result root in DuckDB. Choose any forward_1..5 or reverse_1..5.
SET VARIABLE partition_name = 'forward_1';
CREATE OR REPLACE TEMP VIEW composition_assignments AS
    SELECT unit_id, composition_id, depth
    FROM read_csv('COMPOSITION_ASSIGNMENTS/' || getvariable('partition_name') || '.csv',
        header=true, columns={'unit_id':'BIGINT','composition_id':'BIGINT','depth':'BIGINT'});
CREATE OR REPLACE TEMP VIEW original_nodes AS
    SELECT * FROM 'RELATION_INDEX/nodes.parquet';
CREATE OR REPLACE TEMP VIEW owned_events AS
    SELECT p.*, l.owner, a.composition_id
    FROM 'RELATION_INDEX/event_payload.parquet' p
    JOIN 'EVENT_OWNERSHIP/event_units.parquet' l USING(eid)
    JOIN composition_assignments a ON a.unit_id=l.owner;
CREATE OR REPLACE TEMP VIEW directed_process_relations AS
    SELECT p.*, x.composition_id AS src_composition, y.composition_id AS dst_composition
    FROM 'RELATION_INDEX/process_relations.parquet' p
    JOIN composition_assignments x ON x.unit_id=p.src_uid
    JOIN composition_assignments y ON y.unit_id=p.dst_uid;
CREATE OR REPLACE TEMP VIEW resource_members AS
    SELECT i.segment, i.resource, i.uid, a.composition_id
    FROM 'RELATION_INDEX/resource_incidence.parquet' i
    JOIN composition_assignments a ON i.uid=a.unit_id;
CREATE OR REPLACE TEMP VIEW shared_resource_memberships AS
    SELECT * FROM read_parquet('BOUNDARY_RELATIONS/' || getvariable('partition_name') ||
                               '/shared_resource_groups.parquet');
CREATE OR REPLACE TEMP VIEW boundary_subject_targets AS
    SELECT * FROM read_parquet('BOUNDARY_RELATIONS/' || getvariable('partition_name') ||
                               '/target_subjects.parquet');
CREATE OR REPLACE TEMP VIEW resource_event_refs AS
    SELECT r.*, a.composition_id
    FROM 'RELATION_INDEX/event_resource_refs.parquet' r
    JOIN composition_assignments a ON r.uid=a.unit_id;
-- For a given (segment,resource), resource_members encodes its full clique
-- without asserting producer-consumer causality. Join resource_event_refs to
-- owned_events by eid for operation, direction, endpoints, time and source.
-- To enumerate references to *other* compositions for selected events:
-- SELECT r.eid, r.composition_id AS owner_composition, other.cid AS referred_composition
-- FROM resource_event_refs r JOIN shared_resource_memberships other USING(segment,resource)
-- WHERE r.eid = <selected physical event ID> AND other.cid <> r.composition_id;
