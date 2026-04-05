import json
import os
from typing import Dict, List
from urllib.parse import unquote

import pandas as pd


def _extract_filename_from_audio_uri(audio_uri: str) -> str:
    """Extract filename from Label Studio audio URI/path."""
    if '?d=' in audio_uri:
        _, q = audio_uri.split('?d=', 1)
        path = unquote(q)
    else:
        path = unquote(audio_uri)

    path = path.replace('\\', os.sep).replace('/', os.sep)
    return os.path.basename(path)


def convert_labelstudio_json_to_csv(input_json, output_csv):
    """Convert Label Studio exported JSON annotations to SED CSV format."""
    input_json = unquote(input_json)
    with open(input_json, 'r', encoding='utf-8') as f:
        records = json.load(f)

    rows: List[Dict] = []
    for rec in records:
        audio_uri = rec.get('data', {}).get('audio', '')
        if not audio_uri:
            continue

        filename = _extract_filename_from_audio_uri(audio_uri)

        for ann in rec.get('annotations', []):
            for item in ann.get('result', []):
                value = item.get('value', {})
                start = value.get('start')
                end = value.get('end')
                labels = value.get('labels', [])
                if start is None or end is None or not labels:
                    continue

                for label in labels:
                    rows.append({
                        'filename': filename,
                        'start_time': float(start),
                        'end_time': float(end),
                        'event_label': label
                    })

    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} records to {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Convert Label Studio JSON to SED CSV'
    )
    parser.add_argument(
        '--input', default='data/anno6.json', help='input JSON file path')
    parser.add_argument(
        '--output', default='data/anno6.csv', help='output CSV path')

    args = parser.parse_args()
    convert_labelstudio_json_to_csv(args.input, args.output)
