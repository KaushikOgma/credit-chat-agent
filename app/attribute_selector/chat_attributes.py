from app.utils.config import settings

class ChatProjections:
    """ Projection configurations for user collection queries that support dynamic time zones. """


    @staticmethod
    def get_all_attribute(timezone=None):
        # Use the provided timezone or fall back to the default
        tz = timezone if timezone else settings.APP_TIMEZONE

        return {
            "user_id": "$user_id",  # Renamed
            "credit_service_user_id": "$credit_service_user_id",  # Renamed
            "text": "$content",  # Renamed
            "question_number": "$question_number",  # Renamed
            "sent_by_user": "$sent_by_user",  # Renamed
            "date": {
                "$dateToString": {
                    "format": "%Y-%m-%d",
                    "date": "$timestamp",
                    "timezone": tz
                }
            },  # Renamed
            "time": {
                "$dateToString": {
                    "format": "%H:%M",
                    "date": "$timestamp",
                    "timezone": tz
                }
            },  # Renamed
            "timestamp": {
                "$dateToString": {
                    "format": settings.ACCEPTED_DATE_TIME_STRING,
                    "date": "$timestamp",
                    "timezone": tz
                }
            }
        }