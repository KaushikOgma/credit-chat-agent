from app.utils.config import settings


class UserProjections:
    """ Projection configurations for user collection queries that support dynamic time zones. """

    @staticmethod
    def get_silected_attribute(timezone=None):
        # Use the provided timezone or fall back to the default
        tz = timezone if timezone else settings.APP_TIMEZONE

        return {
            "_id": 1,
            "name": 1,
            "email": 1,
            "contactAddress": 1,
            "contactNo": 1,
            "isActive": 1,
            "isAdmin": 1,
            "config": 1,
            "userId": 1,
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
    def get_all_attribute(timezone=None):
        # Use the provided timezone or fall back to the default
        tz = timezone if timezone else settings.APP_TIMEZONE

        return {
            "_id": 1,
            "name": 1,
            "email": 1,
            "password": 1,
            "contactAddress": 1,
            "contactNo": 1,
            "isActive": 1,
            "isAdmin": 1,
            "config": 1,
            "userId": 1,
            "apiKey": 1,
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