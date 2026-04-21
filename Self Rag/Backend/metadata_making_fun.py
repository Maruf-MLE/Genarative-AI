def metadata_for_teacher(record: dict,metadata:dict) ->dict:
    metadata['name'] = record.get('name')
    metadata['designation'] = record.get('designation')
    metadata['dipertment'] = record.get('department')
    metadata['phone'] = record.get('phone')
    metadata['doc_type'] = record.get('doc_type')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_staff(record:dict,metadata:dict) -> dict:
    metadata['name'] = record.get('name')
    metadata['designation'] = record.get('designation')
    metadata['dipertment'] = record.get('department')
    metadata['email'] = record.get('email')
    metadata['phone'] = record.get('phone')
    metadata['doc_type'] = record.get('doc_type')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_recent_dev(record:dict,metadata:dict) -> dict:
    metadata['doc_type'] = record.get('doc_type')
    metadata['institution'] = record.get('institution')
    metadata['institution_short'] = record.get('institution_short')
    metadata['topic'] = record.get('topic')
    metadata['language'] = record.get('language')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_overvew(record:dict,metadata:dict) -> dict:
    metadata['doc_type'] = record.get('doc_type')
    metadata['institution'] = record.get('institution')
    metadata['institution_short'] = record.get('institution_short')
    metadata['topic'] = record.get('topic')
    metadata['language'] = record.get('language')
    metadata['context_text'] = record.get('context_text')

    return metadata



def metadata_for_mission(record:dict,metadata:dict) -> dict:
    metadata['doc_type'] = record.get('doc_type')
    metadata['institution'] = record.get('institution')
    metadata['institution_short'] = record.get('institution_short')
    metadata['topic'] = record.get('topic')
    metadata['language'] = record.get('language')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_labinfo(record:dict,metadata:dict) -> dict:
    metadata['dipertment'] = record.get('department')
    metadata['doc_type'] = record.get('doc_type')
    metadata['total_labs'] = record.get('total_labs')
    metadata['labs'] = record.get('labs')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_mission(record:dict,metadata:dict) -> dict:
    metadata['doc_type'] = record.get('doc_type')
    metadata['institution'] = record.get('institution')
    metadata['institution_short'] = record.get('institution_short')
    metadata['topic'] = record.get('topic')
    metadata['language'] = record.get('language')
    metadata['context_text'] = record.get('context_text')

    return metadata



def metadata_for_future_dev(record:dict,metadata:dict) -> dict:
    metadata['doc_type'] = record.get('doc_type')
    metadata['institution'] = record.get('institution')
    metadata['institution_short'] = record.get('institution_short')
    metadata['topic'] = record.get('topic')
    metadata['language'] = record.get('language')
    metadata['context_text'] = record.get('context_text')

    return metadata


def metadata_for_class_rutine(record:dict,metadata:dict) -> dict:
    metadata['department'] = record.get('department')
    metadata['semester'] = record.get('semester')
    metadata['session'] = record.get('session')
    metadata['group'] = record.get('group')
    metadata['day'] = record.get('day')
    metadata['doc_type'] = record.get('doc_type')
    metadata['context_text'] = record.get('context_text')


    return metadata
