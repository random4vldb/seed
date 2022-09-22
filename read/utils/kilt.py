def to_distinct_doc_ids(passage_ids):
    doc_ids = set()
    for pid in passage_ids:
        doc_id = pid[:pid.find(':')]
        doc_ids.add(doc_id)
    return list(doc_ids)