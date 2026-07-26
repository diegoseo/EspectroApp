import json
from datetime import datetime

import pytest

from core.analysis_history import AnalysisHistoryEntry, AnalysisHistoryManager
from core.pipeline_manager import PipelineManager


def test_history_entry_round_trip():
    entry = AnalysisHistoryEntry(
        dataset='ftir.csv',
        operation='PCA',
        timestamp=datetime(2026, 7, 17, 10, 30, 0),
        output_dataset='ftir_pca',
        parameters={'components': 3},
        source_datasets=('ftir.csv',),
    )

    restored = AnalysisHistoryEntry.from_dict(entry.to_dict())

    assert restored == entry
    assert restored.timestamp_text == '17/07/2026  10:30:00'


def test_history_manager_add_persist_group_export_and_clear(tmp_path):
    storage = tmp_path / 'analysis_history.json'
    manager = AnalysisHistoryManager(storage_path=storage)

    first = manager.add(
        dataset='ftir.csv',
        operation='Preprocessing',
        output_dataset='ftir_processed',
        parameters={'method': 'area normalization'},
    )
    manager.add(
        dataset='ftir.csv',
        operation='PCA',
        parameters={'components': 3},
        source_datasets=['ftir_processed'],
    )

    assert storage.exists()
    assert len(manager.entries) == 2
    assert manager.grouped_by_dataset()['ftir.csv'][0] == first

    reloaded = AnalysisHistoryManager(storage_path=storage)
    assert len(reloaded.entries) == 2
    assert reloaded.entries[1].source_datasets == ('ftir_processed',)

    json_path = tmp_path / 'exports' / 'history.json'
    csv_path = tmp_path / 'exports' / 'history.csv'
    reloaded.export_json(json_path)
    reloaded.export_csv(csv_path)

    assert json.loads(json_path.read_text(encoding='utf-8'))['entries'][0]['dataset'] == 'ftir.csv'
    assert 'Preprocessing' in csv_path.read_text(encoding='utf-8-sig')

    reloaded.clear()
    assert reloaded.entries == ()
    assert json.loads(storage.read_text(encoding='utf-8'))['entries'] == []


def test_history_rejects_empty_operation_and_survives_invalid_json(tmp_path):
    storage = tmp_path / 'history.json'
    storage.write_text('{broken', encoding='utf-8')
    manager = AnalysisHistoryManager(storage_path=storage)
    assert manager.entries == ()

    with pytest.raises(ValueError, match='cannot be empty'):
        manager.add('dataset', '   ')


def test_pipeline_manager_full_lifecycle(tmp_path):
    manager = PipelineManager(directory=tmp_path / 'pipelines')
    options = {'normalization': 'area', 'derivative': 2}

    saved_name = manager.save(' Raman / Pipeline ', options)
    assert saved_name == 'Raman _ Pipeline'
    assert manager.list_names() == ['Raman _ Pipeline']

    payload = manager.load(saved_name)
    assert payload['pipeline_type'] == 'spectral_preprocessing'
    assert payload['options'] == options

    manager.delete(saved_name)
    assert manager.list_names() == []
    with pytest.raises(FileNotFoundError):
        manager.load(saved_name)


def test_pipeline_manager_validation_and_corrupt_files(tmp_path):
    manager = PipelineManager(directory=tmp_path)

    with pytest.raises(ValueError, match='invalid'):
        manager.save('***', {})

    (tmp_path / 'broken.json').write_text('{bad', encoding='utf-8')
    (tmp_path / 'wrong.json').write_text(
        json.dumps({'name': 'Wrong', 'pipeline_type': 'other', 'options': {}}),
        encoding='utf-8',
    )
    assert manager.list_names() == ['Wrong']

    with pytest.raises(ValueError, match='not a preprocessing pipeline'):
        manager.load('wrong')
