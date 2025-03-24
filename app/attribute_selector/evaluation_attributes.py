from app.utils.config import settings

class EvaluationProjections:
    """ Projection configurations for user collection queries that support dynamic time zones. """


    @staticmethod
    def get_all_attribute(timezone=None):
        # Use the provided timezone or fall back to the default
        tz = timezone if timezone else settings.APP_TIMEZONE

        return {
            "_id": 1,
            "fileName": 1,
            "question": 1,
            "answer": 1,
            "isActive": 1,
            "createdAt": {
                "$dateToString": {
                    "format": settings.ACCEPTED_DATE_TIME_STRING,
                    "date": "$createdAt",
                    'timezone': tz
                }
            },
            "updatedAt": {
                "$dateToString": {
                    "format": settings.ACCEPTED_DATE_TIME_STRING,
                    "date": "$updatedAt",
                    'timezone': tz
                }
            }
        }
    

    @staticmethod
    def get_qa_attribute():
        return {
            "_id": 0,
            "question_id": "$_id",
            "question": 1,
            "answer": 1
        }