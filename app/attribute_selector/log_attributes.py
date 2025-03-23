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
            "logTrail": {
                "$map": {
                    "input": "$logTrail",  # Iterate over logTrail array
                    "as": "log",  # Alias for each element
                    "in": {
                        "message": "$$log.message",
                        "type": "$$log.type",
                        "stackTrace": "$$log.stackTrace",
                        "timestamp": {
                            "$dateToString": {
                                "format": settings.ACCEPTED_DATE_TIME_STRING,
                                "date": "$$log.timestamp",  # Correctly reference timestamp
                                "timezone": tz
                            }
                        }
                    }
                }
            },
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

