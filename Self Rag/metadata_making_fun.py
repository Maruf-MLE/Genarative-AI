def metadata_for_teacher(record: dict,metadata:dict) ->dict:
    metadata['name'] = record.get('name')
    metadata['designation'] = record.get('designation')
    metadata['dipertment'] = record.get('department')
    metadata['phone'] = record.get('phone')
    metadata['doc_type'] = record.get('doc_type')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_staff(record:dict,metadata:dict) -> dict:
    metadata['']