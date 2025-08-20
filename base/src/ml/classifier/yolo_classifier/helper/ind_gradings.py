gradings = {
    '1': 1,
    '2': 3,
    '3': 6,
    '4': 8,
    '91': 11,
    '95': 15,
    '96': 16,
    '97': 17
}

class IndGradings:
    @staticmethod
    def get_db_grading_id(classification_code):
        return gradings[str(classification_code)]
