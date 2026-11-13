"""Map the selected cohort to GDC SVS metadata; never request file contents.

Local case_uuid is GDC cases.case_id; local case_id is cases.submitter_id.
The old slide extraction stops at 2,000 files, so query selected cases with
pagination. Preserve cohort order and emit a blank-file row for unmatched cases.
"""
import csv
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / 'data' / 'metadata'
ENDPOINT = 'https://api.gdc.cancer.gov/files'


def read_csv(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    patients = read_csv(META / 'patients_250.csv')
    local = read_csv(META / 'slides.csv')
    selected = {p['case_uuid']: p['case_id'] for p in patients}
    if (len(selected) != len(patients)
            or len(set(selected.values())) != len(patients)
            or any(not k or not v for k, v in selected.items())):
        raise ValueError('Missing or duplicate patient identifiers')
    for slide in local:
        if slide['case_uuid'] in selected and slide['case_id'] != selected[slide['case_uuid']]:
            raise ValueError('Local UUID/barcode mismatch')
    mapping = defaultdict(dict)
    all_hits = []
    case_ids = list(selected)
    # Small GET batches avoid overlong URLs; each response is independently paginated.
    for start in range(0, len(case_ids), 25):
        batch = case_ids[start:start + 25]
        filters = {'op': 'and', 'content': [
            {'op': '=', 'content': {'field': 'cases.project.project_id', 'value': 'TCGA-BRCA'}},
            {'op': '=', 'content': {'field': 'data_format', 'value': 'SVS'}},
            {'op': 'in', 'content': {'field': 'cases.case_id', 'value': batch}}]}
        offset = 0
        expected_total = None
        while True:
            params = {'filters': json.dumps(filters), 'format': 'JSON', 'size': 100,
                      'from': offset, 'sort': 'file_id:asc',
                      'fields': 'file_id,file_name,data_format,cases.case_id,cases.submitter_id'}
            with urlopen(ENDPOINT + '?' + urlencode(params), timeout=60) as response:
                data = json.load(response)['data']
            hits = data['hits']
            total = data['pagination']['total']
            if expected_total is not None and total != expected_total:
                raise ValueError('GDC result count changed during pagination; rerun')
            expected_total = total
            if not hits and offset < total:
                raise ValueError('Incomplete GDC pagination')
            for hit in hits:
                uuid, name = hit['file_id'], hit['file_name']
                if not uuid or not name.lower().endswith('.svs'):
                    raise ValueError('Invalid SVS metadata')
                matched = False
                for case in hit.get('cases', []):
                    key = case['case_id']
                    if key not in batch:
                        continue
                    matched = True
                    if case['submitter_id'] != selected[key]:
                        raise ValueError('GDC UUID/barcode mismatch')
                    if uuid in mapping[key] and mapping[key][uuid] != name:
                        raise ValueError('Conflicting filenames for file UUID')
                    mapping[key][uuid] = name
                if not matched:
                    raise ValueError('GDC returned a file unrelated to requested cases')
            all_hits.extend(hits)
            offset += len(hits)
            if offset >= total:
                break
        print('Queried patients %d-%d: %d SVS metadata records' %
              (start + 1, start + len(batch), offset), flush=True)
    rows = []
    for p in patients:
        files = mapping[p['case_uuid']]
        for uuid, name in sorted(files.items(), key=lambda item: (item[1], item[0])) or [('', '')]:
            rows.append({'patient_id': p['case_id'], 'case_uuid': p['case_uuid'],
                         'slide_file_uuid': uuid, 'slide_file_name': name,
                         'slide_count_for_patient': len(files)})
    local_pairs = {(s['case_uuid'], s['file_uuid']) for s in local
                   if s['case_uuid'] in selected and s['file_name'].lower().endswith('.svs')}
    api_pairs = {(key, uuid) for key, files in mapping.items() for uuid in files}
    report = {'selected_patients': len(patients),
              'patients_with_svs': sum(bool(mapping[k]) for k in selected),
              'patients_without_svs': sum(not mapping[k] for k in selected),
              'total_svs_files': len({uuid for _, uuid in api_pairs}),
              'patient_slide_associations': len(api_pairs),
              'patients_with_multiple_slides': sum(len(mapping[k]) > 1 for k in selected),
              'additional_associations_vs_local': len(api_pairs - local_pairs),
              'local_associations_not_in_api': len(local_pairs - api_pairs),
              'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'metadata_endpoint': ENDPOINT}
    destination = META / 'patient_slide_mapping.csv'
    with destination.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Validate the saved artifact against the full cohort and distinct file counts.
    saved = read_csv(destination)
    assert len(saved) == len(rows)
    assert {r['case_uuid'] for r in saved} == set(selected)
    for key in selected:
        patient_rows = [r for r in saved if r['case_uuid'] == key]
        files = {r['slide_file_uuid'] for r in patient_rows if r['slide_file_uuid']}
        assert len(files) == len(mapping[key])
        assert all(int(r['slide_count_for_patient']) == len(files) for r in patient_rows)
    (META / 'patient_slide_mapping_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(destination.read_text().splitlines()[0])
    print('\n'.join(destination.read_text().splitlines()[1:11]))


if __name__ == '__main__':
    main()
