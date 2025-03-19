from app.utils.config import settings

class LogProjections:
    """ Projection configurations for log collection queries that support dynamic time zones. """

    @staticmethod
    def get_all_attribute(timezone=None):
        # Use the provided timezone or fall back to the default
        tz = timezone if timezone else settings.APP_TIMEZONE

        return {
            "_id": 1,
            "message": 1,
            "status": 1,
            "logTrail": 1,
            "moduleName": 1,
            "serviceName": 1,
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

