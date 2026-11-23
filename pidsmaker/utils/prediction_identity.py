"""Stable inference batch identity; timestamps remain exact native nanoseconds."""
import re
_SUFFIX=re.compile(r'__g\d{6}_b\d{9}$')
def prediction_name(interval,group,batch):
    return f'{interval}__g{group:06d}_b{batch:09d}'
def timestamp_part(value):
    return _SUFFIX.sub('', value.removesuffix('.csv'))
