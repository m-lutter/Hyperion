"""Regenerate the activity index and calculation input contracts."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from hyperion.verification import activities, catalog, input_contracts

table=activities()
functions,_=catalog()
table['code_route']=table.kernel.map(lambda k:
    f'{functions[k].__module__}.{functions[k].__name__}' if isinstance(k,str)
    else 'hyperion.verification.check_evidence / measured_limits')
table['route_type']=table.kernel.map(lambda k:'calculation' if isinstance(k,str) else 'external evidence validation')
table.to_csv(ROOT/'data/activities.csv',index=False)
input_contracts().to_csv(ROOT/'data/calculation_inputs.csv',index=False)
print(f'Indexed {len(table)} activities; {table.kernel.notna().sum()} have calculation assignments.')
